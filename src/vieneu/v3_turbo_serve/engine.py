"""
Batched serving engine for v3 Turbo (static batching).

Ties the batched backbone + batched acoustic frame generation into one loop that
advances B requests together:

    prefill(prompts) -> loop { acoustic frame (B) -> EOS check -> backbone step (B) }

Finished requests are masked out (their last frame is kept) and the batch keeps
running until all finish or ``max_new_frames``. Each request's codes are decoded
to a waveform at the end, several rows per codec call (``decode_rows``). Pure
PyTorch, runs on CUDA (or CPU).

Concurrent ``generate_batch`` calls (Gradio and servers call ``infer()`` from
several threads) used to run one after another, each as its own small batch.
They now go through ``_Coalescer``: calls that arrive while a run is in
progress wait, and the next run takes every waiting call with the same
sampling settings as one batch.
"""
from __future__ import annotations

import logging
import math
import os
import re
import threading
from collections import OrderedDict
from typing import List, Optional

import numpy as np
import torch

from .batched_acoustic import generate_frame_batched
from .batched_backbone import BatchedBackbone
from .._v3_turbo_engine.rep_history import DEFAULT_REP_WINDOW, RepetitionHistory

logger = logging.getLogger("Vieneu.V3TurboServe")

# Guard chống "chunk câm": một row khỏe sinh ≥ ~0.4 frame trên mỗi ký tự phoneme
# (đo thực nghiệm 0.41–1.0); dưới 0.25 gần như chắc chắn là EOS sớm / cụt giữa
# chừng → sinh lại row đó. Độ dài phoneme được tính SAU khi loại các tag markup
# (<en>…</en>, <|emotion_k|>) vì chúng chiếm ký tự nhưng không tốn frame audio.
MIN_FRAMES_PER_PHONE = 0.25
_MARKUP_RE = re.compile(r"<\|emotion_\d+\|>|</?en>")


def _min_expected_frames(phonemes: str) -> int:
    eff_len = len(_MARKUP_RE.sub("", phonemes or ""))
    return max(3, math.ceil(MIN_FRAMES_PER_PHONE * eff_len))


# Chặn-trên đối xứng: trần frame theo độ dài phoneme (24 + 2.0/ký tự, đo trên
# dataset finetune — xem vieneu_utils.core_utils.max_expected_frames). Row ngắn
# bắn trượt stop token thì bị cắt tại trần của CHÍNH row đó thay vì chạy hết
# max_new_frames của cả batch.
from vieneu_utils.core_utils import max_expected_frames, BABBLE_MAX_RETRIES, babble_suspect, babble_prefer, check_sampling


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except ValueError:
        return default


# Codec tokens per code frame in its last (400 Hz) decoder stage: the
# non-streaming decode builds a (tokens x tokens) attention mask per row.
_CODEC_TOKENS_PER_FRAME = 32


def decode_rows(tokenizer, codes: List[torch.Tensor], budget_mb: int = 1024,
                full_codec=None) -> List[np.ndarray]:
    """Mono float32 waveforms for many rows' codes ``(T_i, n_vq)``.

    Same codec decode as one ``decode()`` per row (padding is masked, the
    decoder is causal), but several rows per call. Rows are grouped longest
    first so a group pads little, and a group stays under ``budget_mb`` of
    attention mask (counted at 4 bytes per element; a 300-frame row is
    ~370 MB, a 75-frame row ~23 MB).

    ``full_codec`` (a fast ``codec_stream.StreamCodec``) decodes through
    ``StreamCodec.decode_full`` instead: same audio (exact fp32, where the
    codec's own decode runs its quantizer convs in TF32), 2-4x faster. Its
    groups are capped at ``2 * budget_mb`` padded frames (~0.4 MB of
    activations per frame).
    """
    out: List[Optional[np.ndarray]] = [None] * len(codes)
    todo = []
    for i, c in enumerate(codes):
        if len(c):
            todo.append(i)
        else:
            out[i] = np.zeros(0, dtype=np.float32)
    todo.sort(key=lambda i: -len(codes[i]))
    if full_codec is not None:
        max_frames = max(1, 2 * int(budget_mb))
        k = 0
        while k < len(todo):
            n = max(1, max_frames // len(codes[todo[k]]))
            group = todo[k:k + n]
            k += len(group)
            for i, wav in zip(group, full_codec.decode_full([codes[i] for i in group])):
                out[i] = wav.numpy().copy()
        return out
    dev = next(tokenizer.parameters()).device
    budget = max(1, int(budget_mb)) * 2**20
    k = 0
    while k < len(todo):
        per_row = (len(codes[todo[k]]) * _CODEC_TOKENS_PER_FRAME) ** 2 * 4
        n = max(1, budget // per_row)
        group = todo[k:k + n]
        k += len(group)
        dec = tokenizer.batch_decode([codes[i].T.to(dev) for i in group])
        lens = dec.audio_lengths.tolist()
        audio = dec.audio.float().mean(1).cpu().numpy()
        for j, i in enumerate(group):
            out[i] = audio[j, :lens[j]].copy()
    return out


class _Call:
    def __init__(self, key, items):
        self.key, self.items = key, items
        self.result = None
        self.error: Optional[BaseException] = None
        self.done = False


class _Coalescer:
    """Runs ``fn(items) -> results`` (one result per item, same order) for
    concurrent callers. While a run is in progress new calls wait; the next
    run takes the oldest waiting call plus every other waiting call with the
    same key, up to ``max_items`` items. Each call gets its own slice back.

    If a merged run raises an ``Exception``, its calls are re-run one by one,
    so a caller's bad input fails that caller only.
    """

    def __init__(self, max_items: int):
        self.max_items = max(1, int(max_items))
        self._cv = threading.Condition()
        self._queue: List[_Call] = []
        self._busy = False
        self.runs = 0   # runs made (tests / stats)

    def run(self, key, items, fn):
        call = _Call(key, list(items))
        with self._cv:
            self._queue.append(call)
        while True:
            with self._cv:
                while self._busy and not call.done:
                    self._cv.wait()
                if call.done:
                    break
                batch = self._take()
                self._busy = True
            try:
                self._execute(batch, fn)
            finally:
                with self._cv:
                    for c in batch:
                        c.done = True
                    self._busy = False
                    self._cv.notify_all()
        if call.error is not None:
            raise call.error
        return call.result

    def _take(self) -> List[_Call]:
        first = self._queue[0]
        batch, n = [], 0
        for c in list(self._queue):
            if c.key != first.key:
                continue
            if batch and n + len(c.items) > self.max_items:
                break
            batch.append(c)
            n += len(c.items)
        for c in batch:
            self._queue.remove(c)
        return batch

    def _execute(self, batch: List[_Call], fn) -> None:
        try:
            self.runs += 1
            out = fn([x for c in batch for x in c.items])
        except Exception as e:   # noqa: BLE001 — isolate the caller whose input broke it
            if len(batch) == 1:
                batch[0].error = e
                return
            for c in batch:
                try:
                    self.runs += 1
                    c.result = fn(c.items)
                except Exception as e1:   # noqa: BLE001
                    c.error = e1
            return
        except BaseException as e:   # interrupts / cancel hooks: everyone in the run hears it
            for c in batch:
                c.error = e
            raise
        i = 0
        for c in batch:
            c.result = out[i:i + len(c.items)]
            i += len(c.items)


class V3TurboBatchEngine:
    def __init__(self, tts):
        # tts: VieNeuTTSv3Turbo (provides prompt building, embeddings, heads, codec)
        self.tts = tts
        self.model = tts.model
        self.config = tts.config
        self.bb = BatchedBackbone(tts.model)
        self._graphs = {}  # (B, temp, top_k, top_p) -> CudaGraphedFrame (acoustic step)
        # Whole-frame CUDA graphs (see fused.py), keyed by batch bucket, cache
        # length bucket and sampling settings. The default on CUDA;
        # VIENEU_FUSED_FRAME=0 falls back to the plain per-op loop.
        self._fused = {}
        self.use_fused = os.environ.get("VIENEU_FUSED_FRAME", "1") != "0"
        # fastpath.py on top of the fused graphs (fewer kernels per frame, graphed
        # prefill). VIENEU_FASTPATH=0 keeps fused.py's frame and the HF prefill.
        from . import fastpath
        self.use_fastpath = self.use_fused and fastpath.enabled()
        self._full_codec_obj = None   # see _full_codec
        # Every run overwrites its graph's buffers (static KV cache, codes, flags),
        # so runs must not overlap — infer() is called from several threads by the
        # Gradio app and by servers.
        self._fused_lock = threading.Lock()
        # Each graph pins its own static KV cache (hundreds of MB at batch 32), and
        # a new sampling setting (e.g. a temperature slider) means new graphs: only
        # the most recently used ``fused_samplings`` settings keep theirs.
        self._fused_lru: "OrderedDict[tuple, None]" = OrderedDict()
        self.fused_samplings = max(1, _env_int("VIENEU_FUSED_SAMPLINGS", 2))
        # Called once per generated frame (both loops). A server sets it to
        # its cancel check; raising from it stops the batch within a frame.
        self.frame_hook = None
        # Concurrent generate_batch calls share runs (module doc). The owner sets
        # ``coalescer.max_items`` to its batch-size ceiling.
        self.coalescer = _Coalescer(_env_int("VIENEU_MAX_BATCH_ROWS", 32))
        # Attention-mask budget of one batched codec decode (decode_rows).
        self.decode_budget_mb = _env_int("VIENEU_DECODE_BUDGET_MB", 1024)

    def _full_codec(self):
        """The fast whole-utterance codec decode (``StreamCodec.decode_full``),
        built on first use; ``None`` off the fast path or for a codec layout
        it does not know (``decode_rows`` then uses the codec's own decode)."""
        if not (self.use_fastpath and self.tts.device.type == "cuda"):
            return None
        if self._full_codec_obj is None:
            from .codec_stream import StreamCodec, UnsupportedCodec
            try:
                codec = StreamCodec(self.tts.audio_tokenizer, 1, use_graph=False)
            except UnsupportedCodec as e:
                logger.warning("v3 Turbo: fast codec decode unavailable (%s)", e)
                codec = None
            self._full_codec_obj = codec if codec is not None and codec.fast else False
        return self._full_codec_obj or None

    def _get_graph(self, B, temperature, top_k, top_p):
        key = (B, round(temperature, 4), top_k, round(top_p, 4))
        if key not in self._graphs:
            from .cudagraph import CudaGraphedFrame
            self._graphs[key] = CudaGraphedFrame(
                self.model, B, temperature=temperature, top_k=top_k, top_p=top_p
            )
        return self._graphs[key]

    @torch.no_grad()
    def _prompt_embeds(self, req) -> torch.Tensor:
        # Each request carries its own reference (speaker_emb + ref_codes) and optional
        # use_ref_codes switch. The speaker anchor is added to every prefill row.
        # A "style" key in the request is accepted but ignored (deprecated): the style
        # is implied by the reference, so the head token is always the natural style.
        style_id = self.tts._resolve_style_id()
        phonemes = req.get("phonemes")
        if phonemes is None:
            from vieneu_utils.phonemize_text import phonemize_text_with_emotions
            phonemes = phonemize_text_with_emotions(req["text"])
        ref_codes = req.get("ref_codes") if req.get("use_ref_codes", True) else None
        spk = self.tts._resolve_speaker_emb(req.get("speaker_emb"))
        prompt_2d = self.tts._build_prompt_2d(phonemes, None, ref_codes, style_id)
        return self.model._build_inputs_embeds(prompt_2d.unsqueeze(0).to(self.tts.device), speaker_emb=spk)[0]  # (T, H)

    def _prompt_ids(self, req):
        """``(ids, spk)`` for the fast path: the prompt as rows of
        ``fastpath.Prepared.emb_table`` ids on the host (``(T, n_vq+1)``), and
        the speaker anchor ``(1, D)`` on the device (or ``None``). Same prompt
        as ``_prompt_embeds``; the embedding happens inside the prefill graph."""
        from .fastpath import Prepared
        phonemes = req.get("phonemes")
        if phonemes is None:
            from vieneu_utils.phonemize_text import phonemize_text_with_emotions
            phonemes = phonemize_text_with_emotions(req["text"])
        ref_codes = req.get("ref_codes") if req.get("use_ref_codes", True) else None
        spk = self.tts._resolve_speaker_emb(req.get("speaker_emb"))
        prompt_2d = self.tts._build_prompt_2d(phonemes, None, ref_codes, self.tts._resolve_style_id())
        return Prepared.of(self.model).prompt_ids(prompt_2d), spk

    @torch.no_grad()
    def generate_batch(
        self,
        requests: List[dict],
        *,
        temperature: float = 0.8,
        top_k: int = 25,
        top_p: float = 0.95,
        repetition_penalty: float = 1.2,
        repetition_window: int = DEFAULT_REP_WINDOW,
        max_new_frames: int = 300,
        use_cudagraph: bool = False,
        max_retries: int = 2,
        frame_cap: bool = True,
    ) -> List[np.ndarray]:
        """Generate a waveform for each request in one batched run.

        ``temperature`` is a scalar applied to every codebook. ``repetition_penalty``
        (default 1.2, matching the single-path engine) down-weights codes already
        produced per row/codebook to avoid repetition artifacts.

        Rows that come back suspiciously short (fewer than ``MIN_FRAMES_PER_PHONE``
        frames per phoneme char — an early-EOS "silent chunk") are regenerated as a
        smaller batch, up to ``max_retries`` times, keeping the longest attempt.
        Set ``max_retries=0`` to disable the guard.

        ``frame_cap=True`` (default) additionally caps each row at its own
        ``max_expected_frames(phonemes)`` — the upper-bound twin of the guard
        above: a short row that misses its stop token gets truncated at a
        length plausible for its text instead of babbling to ``max_new_frames``.

        ``use_cudagraph=True`` captures (and caches) a CUDA graph of the per-frame
        acoustic step for this batch size — big per-step speedup, reused across calls.
        It is ignored when ``repetition_penalty != 1.0`` (the penalty needs dynamic
        per-row history, which a static graph cannot hold).
        """
        # Before any GPU work, in the caller's thread: a NaN baked into a graph is
        # a device-side assert, and it must fail this call, not a merged run.
        temperature, top_k, top_p, repetition_penalty = check_sampling(
            temperature, top_k, top_p, repetition_penalty)
        # Fill in phonemes up front so the guard can size-check every row (and so
        # retries don't re-phonemize). Each row keeps its call's frame budget, so
        # calls with different budgets can share a run.
        reqs = []
        for r in requests:
            if r.get("phonemes") is None:
                from vieneu_utils.phonemize_text import phonemize_text_with_emotions
                r = {**r, "phonemes": phonemize_text_with_emotions(r["text"])}
            reqs.append({**r, "_max_new_frames": int(max_new_frames)})
        if not reqs:
            return []
        key = (round(temperature, 4), int(top_k), round(top_p, 4), round(repetition_penalty, 4),
               int(repetition_window), bool(use_cudagraph), int(max_retries), bool(frame_cap))
        return self.coalescer.run(key, reqs, lambda rows: self._run_batch(
            rows, temperature=temperature, top_k=top_k, top_p=top_p,
            repetition_penalty=repetition_penalty, repetition_window=repetition_window,
            use_cudagraph=use_cudagraph, max_retries=max_retries, frame_cap=frame_cap,
        ))

    @torch.no_grad()
    def _run_batch(self, reqs: List[dict], *, temperature, top_k, top_p, repetition_penalty,
                   repetition_window, use_cudagraph, max_retries, frame_cap) -> List[np.ndarray]:
        """One run of ``generate_batch`` over ``reqs`` (possibly several calls')."""
        row_max = [r["_max_new_frames"] for r in reqs]
        sampling = dict(
            temperature=temperature, top_k=top_k, top_p=top_p,
            repetition_penalty=repetition_penalty, repetition_window=repetition_window,
            max_new_frames=max(row_max), use_cudagraph=use_cudagraph,
        )
        # Trần frame RIÊNG từng row (luôn > floor chặn-dưới nên không kích retry).
        caps = [min(m, max_expected_frames(r["phonemes"])) if frame_cap else m
                for m, r in zip(row_max, reqs)]
        codes = self._generate_codes_batch(reqs, frame_caps=caps, **sampling)

        # Guard: re-run rows whose output is too short for their text.
        floors = [_min_expected_frames(r["phonemes"]) for r in reqs]
        for _ in range(max(0, max_retries)):
            bad = [i for i in range(len(reqs)) if len(codes[i]) < floors[i]]
            if not bad:
                break
            logger.warning(
                f"⚠️ v3 Turbo batch: {len(bad)} chunk ngắn bất thường "
                f"(rows {bad}) — đang sinh lại...")
            retry = self._generate_codes_batch(
                [reqs[i] for i in bad], frame_caps=[caps[i] for i in bad], **sampling)
            for j, i in enumerate(bad):
                if len(retry[j]) > len(codes[i]):
                    codes[i] = retry[j]
        else:
            still_bad = [i for i in range(len(reqs)) if len(codes[i]) < floors[i]]
            if still_bad and max_retries > 0:
                logger.warning(
                    f"⚠️ v3 Turbo batch: rows {still_bad} vẫn ngắn hơn ngưỡng sau "
                    f"{max_retries} lần thử — giữ bản dài nhất.")

        def _decode(rows) -> List[np.ndarray]:
            return decode_rows(self.tts.audio_tokenizer, rows, self.decode_budget_mb,
                               full_codec=self._full_codec())

        wavs: List[np.ndarray] = _decode(codes)

        # Babble guard (đối xứng với guard "chunk câm" ở trên): row rất ngắn mà
        # "nói thêm" sau stop token trượt -> sinh lại row đó. Đây là tầng engine
        # nên mọi lối vào (SDK, Gradio, server) đều được che. Xem core_utils.babble_suspect.
        retries = int(getattr(self, "babble_retries", BABBLE_MAX_RETRIES))
        if retries > 0 and frame_cap:
            sr = int(getattr(self.tts, "sample_rate", 48_000))
            state = [babble_suspect(wavs[i], sr, reqs[i]["phonemes"], caps[i], len(codes[i]))
                     for i in range(len(reqs))]
            for attempt in range(retries):
                bad = [i for i in range(len(reqs)) if state[i][0]]
                if not bad:
                    break
                logger.info(f"🔁 v3 Turbo batch: {len(bad)} chunk ngắn nghi 'nói thêm' (rows {bad}) "
                            f"— đang sinh lại ({attempt + 1}/{retries})...")
                retry = self._generate_codes_batch(
                    [reqs[i] for i in bad], frame_caps=[caps[i] for i in bad], **sampling)
                retry_wavs = _decode(retry)
                for j, i in enumerate(bad):
                    w2 = retry_wavs[j]
                    cand = babble_suspect(w2, sr, reqs[i]["phonemes"], caps[i], len(retry[j]))
                    if babble_prefer(cand, state[i]):
                        codes[i], wavs[i], state[i] = retry[j], w2, cand
        return wavs

    @torch.no_grad()
    def _generate_codes_batch(
        self,
        requests: List[dict],
        *,
        temperature: float = 0.8,
        top_k: int = 25,
        top_p: float = 0.95,
        repetition_penalty: float = 1.2,
        repetition_window: int = DEFAULT_REP_WINDOW,
        max_new_frames: int = 300,
        use_cudagraph: bool = False,
        frame_caps: Optional[List[int]] = None,
    ) -> List[torch.Tensor]:
        """One batched generation pass; returns per-row codes ``(T, n_vq)`` (no decode).

        ``frame_caps[b]`` (nếu có) là trần frame riêng của row ``b`` — chạm trần
        thì row đó coi như xong (mask ra khỏi batch) dù chưa thấy EOS.
        """
        cfg = self.config
        n_vq = cfg.n_vq
        eos_id = cfg.speech_generation_end_token_id
        sgs = cfg.speech_generation_start_token_id
        pad = cfg.audio_pad_token_id
        dev = self.tts.device

        if self.use_fastpath and dev.type == "cuda":
            return self._generate_codes_fastpath(
                requests, frame_caps, temperature=temperature, top_k=top_k, top_p=top_p,
                repetition_penalty=repetition_penalty, repetition_window=repetition_window,
                max_new_frames=max_new_frames,
            )

        embeds_list = [self._prompt_embeds(r) for r in requests]
        B = len(requests)

        # Per-request speaker anchor, stacked to (B, D), re-added at every decode step
        # (broadcast over the single new row). None when the model has no speaker encoder.
        spk_list = [self.tts._resolve_speaker_emb(r.get("speaker_emb")) for r in requests]
        batch_spk = torch.cat(spk_list, dim=0) if (spk_list and spk_list[0] is not None) else None

        if self.use_fused and dev.type == "cuda":
            return self._generate_codes_fused(
                embeds_list, batch_spk, frame_caps,
                temperature=temperature, top_k=top_k, top_p=top_p,
                repetition_penalty=repetition_penalty, repetition_window=repetition_window,
                max_new_frames=max_new_frames,
            )

        # Per-row, per-codebook sliding-window history for the repetition penalty
        # (matches the single-path decode_one_frame). None when the penalty is disabled.
        history = ([RepetitionHistory(n_vq, repetition_window) for _ in range(B)]
                   if not math.isclose(repetition_penalty, 1.0) else None)
        # CUDA graph bakes in a static step → incompatible with dynamic rep-penalty.
        use_graph = use_cudagraph and dev.type == "cuda" and math.isclose(repetition_penalty, 1.0)
        graphed = self._get_graph(B, temperature, top_k, top_p) if use_graph else None

        h, cache, mask, pos = self.bb.prefill(embeds_list)   # h: (B, H)
        finished = [False] * B
        codes_per_req: List[List[torch.Tensor]] = [[] for _ in range(B)]

        for _ in range(max_new_frames):
            if self.frame_hook is not None:
                self.frame_hook()
            if graphed is not None:
                codes, is_eos = graphed.run(h)               # acoustic frame via CUDA graph
            else:
                codes, prefill_out = generate_frame_batched(
                    self.model, h, temperature=temperature, top_k=top_k, top_p=top_p,
                    repetition_penalty=repetition_penalty, history=history,
                )
                is_eos = (self.model.text_lm_head(prefill_out[:, 0]).float().argmax(-1) == eos_id)
            for b in range(B):
                if not finished[b]:
                    codes_per_req[b].append(codes[b])   # include the EOS frame (matches single path)
                    if bool(is_eos[b]) or (
                        frame_caps is not None and len(codes_per_req[b]) >= frame_caps[b]
                    ):
                        finished[b] = True
            if all(finished):
                break

            # Feed the generated frame back as the next backbone input.
            slot = torch.full((B, 1, n_vq + 1), pad, dtype=torch.long, device=dev)
            slot[:, :, 0] = sgs
            slot[:, 0, 1:] = codes
            se = self.model._build_inputs_embeds(slot, speaker_emb=batch_spk)
            h, cache, mask, pos = self.bb.decode_step(se, cache, mask, pos)

        return [
            torch.stack(codes_per_req[b]) if codes_per_req[b]
            else torch.zeros(0, cfg.n_vq, dtype=torch.long)
            for b in range(B)
        ]

    # Frames a fused graph's code buffer holds; a call may ask for fewer.
    FUSED_MAX_FRAMES = 512

    def warm_fused(self, batch_sizes=(1, 16), max_len: int = 1024, *,
                   temperature: float = 0.8, top_k: int = 25, top_p: float = 0.95,
                   repetition_penalty: float = 1.2,
                   repetition_window: int = DEFAULT_REP_WINDOW) -> int:
        """Capture the fused graphs a server will need before its first request.

        A capture costs ~0.5 s of warm-up frames, paid on the first call for
        each (batch bucket, cache length, sampling) otherwise. ``max_len`` is
        the cache-length bucket (prompt + frames; 1024 covers a 256-char chunk
        with a 30 s reference). Returns how many graphs were captured.
        """
        from .fused import FusedFrame, batch_bucket, cache_len_bucket

        if not (self.use_fused and self.tts.device.type == "cuda"):
            return 0
        if self.use_fastpath:
            from .fastpath import Prefiller, Prepared
            n = 0
            with self._fused_lock:
                for b in batch_sizes:
                    # Every cache bucket up to ``max_len``: short chunks land in 512.
                    for needed in range(512, max(512, int(max_len)) + 1, 512):
                        key = self._fastpath_key(int(b), needed, temperature, top_k, top_p,
                                                 repetition_penalty, repetition_window)
                        n += key not in self._fused
                        self._fastpath_frame(key)
            # Prompt buckets up to 512 tokens (a 256-char chunk + a long reference).
            Prefiller.of(Prepared.of(self.model)).warm(512)
            return n
        max_len = cache_len_bucket(max_len, int(self.config.max_position_embeddings))
        n = 0
        with self._fused_lock:
            for b in batch_sizes:
                bp = batch_bucket(int(b))
                key = (bp, max_len, round(temperature, 4), int(top_k), round(top_p, 4),
                       round(repetition_penalty, 4), int(repetition_window))
                if key not in self._fused:
                    n += 1
                self._fused_frame(key, lambda bp=bp: FusedFrame(
                    self.model, bp, max_len, self.FUSED_MAX_FRAMES,
                    temperature=temperature, top_k=top_k, top_p=top_p,
                    repetition_penalty=repetition_penalty, repetition_window=repetition_window,
                ))
        return n

    def _fused_frame(self, key: tuple, create):
        """The cached graph for ``key``; ``create()`` captures it on first use.

        Call with ``_fused_lock`` held. Using a sampling setting (``key[2:]``)
        makes it the most recent; the graphs of older settings beyond
        ``fused_samplings`` are dropped before a new capture allocates.
        """
        sampling = key[2:]
        self._fused_lru[sampling] = None
        self._fused_lru.move_to_end(sampling)
        while len(self._fused_lru) > self.fused_samplings:
            old, _ = self._fused_lru.popitem(last=False)
            for k in [k for k in self._fused if k[2:] == old]:
                del self._fused[k]
            logger.info("v3 Turbo: freed the CUDA graphs of sampling %s (keeping %d settings; "
                        "VIENEU_FUSED_SAMPLINGS)", old, self.fused_samplings)
        fused = self._fused.get(key)
        if fused is None:
            fused = self._fused[key] = create()
        return fused

    @torch.no_grad()
    def _generate_codes_fused(
        self, embeds_list, batch_spk, frame_caps, *, temperature, top_k, top_p,
        repetition_penalty, repetition_window, max_new_frames,
    ) -> List[torch.Tensor]:
        """``_generate_codes_batch`` through one captured CUDA graph per frame.

        The batch is padded to a power of two (repeating the last row) and the
        KV cache length rounded up to a bucket, so a handful of graphs serve
        every call; a graph costs a few warm-up frames to capture and is kept.
        """
        from .fused import FusedFrame, batch_bucket, cache_len_bucket

        cfg = self.config
        B = len(embeds_list)
        Bp = batch_bucket(B)
        pad = Bp - B
        embeds_p = list(embeds_list) + [embeds_list[-1]] * pad
        max_new_frames = min(int(max_new_frames), self.FUSED_MAX_FRAMES)
        caps = [min(int(c), max_new_frames) for c in (frame_caps or [max_new_frames] * B)]
        caps += [caps[-1]] * pad
        spk_p = None
        if batch_spk is not None:
            spk_p = torch.cat([batch_spk, batch_spk[-1:].expand(pad, -1)], dim=0) if pad else batch_spk
        T = max(e.shape[0] for e in embeds_p)
        max_len = cache_len_bucket(T + max_new_frames + 1, int(cfg.max_position_embeddings))
        key = (Bp, max_len, round(temperature, 4), int(top_k), round(top_p, 4),
               round(repetition_penalty, 4), int(repetition_window))
        with self._fused_lock:
            fused = self._fused_frame(key, lambda: FusedFrame(
                self.model, Bp, max_len, self.FUSED_MAX_FRAMES,
                temperature=temperature, top_k=top_k, top_p=top_p,
                repetition_penalty=repetition_penalty, repetition_window=repetition_window,
            ))
            h, cache, mask, pos = self.bb.prefill(embeds_p)
            codes = fused.run(h, cache, mask, pos, caps, spk_p, max_new_frames, on_frame=self.frame_hook)
        return codes[:B]   # run() returns clones, safe to use after the lock

    def _fastpath_key(self, B: int, needed: int, temperature, top_k, top_p,
                      repetition_penalty, repetition_window) -> tuple:
        """Graph key for ``B`` rows needing a ``needed``-token cache (prompt + frames + 1)."""
        from .fused import batch_bucket, cache_len_bucket
        max_len = cache_len_bucket(needed, int(self.config.max_position_embeddings))
        # key[2:] is the sampling setting the graph LRU (_fused_frame) groups by.
        return (batch_bucket(B), max_len, round(temperature, 4), int(top_k), round(top_p, 4),
                round(repetition_penalty, 4), int(repetition_window), "fast")

    def _fastpath_frame(self, key: tuple):
        from .fastpath import FastBatchFrame, Prepared
        Bp, max_len, temperature, top_k, top_p, repetition_penalty, repetition_window, _ = key
        return self._fused_frame(key, lambda: FastBatchFrame(
            Prepared.of(self.model), Bp, max_len, self.FUSED_MAX_FRAMES,
            temperature=temperature, top_k=top_k, top_p=top_p,
            repetition_penalty=repetition_penalty, repetition_window=repetition_window,
        ))

    @torch.no_grad()
    def _generate_codes_fastpath(
        self, requests, frame_caps, *, temperature, top_k, top_p,
        repetition_penalty, repetition_window, max_new_frames,
    ) -> List[torch.Tensor]:
        """``_generate_codes_batch`` through ``fastpath.FastBatchFrame``: each
        prompt is prefilled by a graph straight into its slot (no HF prefill,
        no padded batch), spare slots of the power-of-two batch stay idle."""
        prompts = [self._prompt_ids(r) for r in requests]
        B = len(prompts)
        max_new_frames = min(int(max_new_frames), self.FUSED_MAX_FRAMES)
        caps = [min(int(c), max_new_frames) for c in (frame_caps or [max_new_frames] * B)]
        T = max(int(ids.shape[0]) for ids, _ in prompts)
        key = self._fastpath_key(B, T + max_new_frames + 1, temperature, top_k, top_p,
                                 repetition_penalty, repetition_window)
        with self._fused_lock:
            frame = self._fastpath_frame(key)
            return frame.run([ids for ids, _ in prompts], [spk for _, spk in prompts], caps,
                             max_new_frames, on_frame=self.frame_hook)
