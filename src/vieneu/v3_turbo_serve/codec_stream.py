"""
Streaming MOSS codec decode over B slots: ring KV cache, one CUDA graph.
========================================================================

Why
---
In ``stream.py`` the codec is most of the cost: on an RTX 3060 one
``batch_decode(streaming=True)`` call took ~65 ms + ~2.5 ms per reserved
slot, against 7-10 ms for an AR frame. Reading the codec
(OpenMOSS-Team/MOSS-Audio-Tokenizer-Nano, revision 6aa02b0) explains both:

* it runs eagerly — a few hundred small kernels per code frame, plus one
  ``.item()`` per row per frame in ``MossAudioTokenizerDecodeSession.step``;
* its streaming attention rebuilds every layer's whole sliding window each
  frame (``torch.cat`` + slice + ``torch.where`` + ``copy_``) for every
  reserved slot: ~25 MB a slot in fp32, ~10 passes a frame.

Its own ``use_cuda_graph`` mode is not usable: ``step()`` keeps views of the
graph's static output buffer, so a K-frame call returns the last frame K
times, and the frame that triggers a capture is returned without being
computed (the session does not replay after capturing).

What
----
``StreamCodec`` decodes one code frame for B slots with the codec's own
modules (quantizer, projections, norms, FFNs, rope) and replaces only the
streaming attention cache: per layer a ring of ``context + T`` positions
(``T`` = tokens the stage gets per code frame), written in place at the
slot's offset. A query at position ``p`` sees exactly the keys of the
sliding window, ``(p - context, p]``, so the audio matches the codec's own
decode up to summation order. The frame is a fixed sequence of device ops:
on CUDA it is captured once and replayed.

A slot that has no frame this step (``exec`` False) is computed anyway —
its output is dropped and its offset does not move. Its ring write lands on
positions that are already out of every future query's window, and the
slot's next real frame overwrites the same indices, so nothing leaks.

``compute_dtype`` (CUDA only) runs the decoder under autocast like the
codec's own ``set_compute_dtype``; the quantizer stays fp32.

``overlap=True`` (CUDA only) puts all codec work on a CUDA stream of its own:
``launch_block`` returns at once and the caller's stream keeps running
(``stream.py`` keeps generating frames meanwhile); ``CodecBlock.result()``
waits for the audio.
"""
from __future__ import annotations

import contextlib
from typing import List, Optional

import torch
import torch.nn.functional as F

SAMPLES_PER_FRAME = 3840   # per channel: 48 kHz / 12.5 frames per second


class UnsupportedCodec(RuntimeError):
    """The loaded codec does not have the layout ``StreamCodec`` was written for."""


class CodecBlock:
    """Audio of one ``launch_block`` call: ``(F, B, 3840)`` float32 on the host."""

    def __init__(self, done: Optional["torch.cuda.Event"], audio: torch.Tensor):
        self._done, self._audio = done, audio

    def ready(self) -> bool:
        return self._done is None or self._done.query()

    def result(self) -> torch.Tensor:
        if self._done is not None:
            self._done.synchronize()
            self._audio = self._audio.clone()   # off the codec's reused pinned buffer
            self._done = None
        return self._audio


class _Stage:
    """One projected transformer of the decoder plus its ring KV caches."""

    def __init__(self, module, T: int, B: int, device, dtype):
        tr = getattr(module, "transformer", None)
        layers = list(getattr(tr, "layers", []))
        if not layers or getattr(tr, "positional_embedding", None) != "rope":
            raise UnsupportedCodec("decoder transformer without rope layers")
        a0 = layers[0].self_attn
        if a0.context is None or not a0.causal:
            raise UnsupportedCodec("decoder attention must be causal with a finite context")
        self.module = module
        self.layers = layers
        self.H = int(a0.num_heads)
        self.E = int(a0.embed_dim)
        self.D = self.E // self.H
        self.ctx = int(a0.context)
        self.T = int(T)
        self.R = self.ctx + self.T
        self.k = torch.zeros(len(layers), B, self.H, self.R, self.D, device=device, dtype=dtype)
        self.v = torch.zeros_like(self.k)
        # Tokens this stage has seen per slot (= the rope offset of the next token).
        self.offset = torch.zeros(B, device=device, dtype=torch.long)
        self._t = torch.arange(self.T, device=device)
        self._r = torch.arange(self.R, device=device)

    def forward(self, x: torch.Tensor, exec_mask: torch.Tensor) -> torch.Tensor:
        """``x`` (B, C, T) conv layout → (B, C', T); advances the ``exec`` slots."""
        m = self.module
        x = m.input_proj(x.transpose(1, 2))                       # (B, T, E)
        B = x.shape[0]
        pos_q = self.offset.view(-1, 1) + self._t                 # (B, T)
        end = pos_q[:, -1:]
        # Ring index j holds the latest position p <= end with p = j (mod R).
        pos_k = end - torch.remainder(end - self._r, self.R)      # (B, R)
        delta = pos_q.unsqueeze(-1) - pos_k.unsqueeze(1)          # (B, T, R)
        mask = ((pos_k >= 0).unsqueeze(1) & (delta >= 0) & (delta < self.ctx)).unsqueeze(1)
        idx = torch.remainder(pos_q, self.R).view(B, 1, self.T, 1).expand(B, self.H, self.T, self.D)
        for i, layer in enumerate(self.layers):
            x = self._layer(i, layer, x, mask, idx)
        self.offset.copy_(torch.where(exec_mask, self.offset + self.T, self.offset))
        return m.output_proj(x).transpose(1, 2)

    def _layer(self, i: int, layer, x: torch.Tensor, mask: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        # MossAudioTokenizerTransformerLayer.forward with our attention core.
        a = layer.self_attn
        residual = x
        h = layer.norm1(x)
        q, k, v = a._project_qkv(h)                               # (B, H, T, D)
        q, k = a.rope(q, k, self.offset, time_before_heads=False)
        kc, vc = self.k[i], self.v[i]
        kc.scatter_(2, idx, k.to(kc.dtype))
        vc.scatter_(2, idx, v.to(vc.dtype))
        att = F.scaled_dot_product_attention(q, kc, vc, mask, dropout_p=0.0)
        att = att.transpose(1, 2).reshape(x.shape[0], self.T, self.E)
        x = residual.to(h) + layer.layer_scale_1(a.out_proj(att))
        residual = x
        h = layer.norm2(x)
        return residual.to(h) + layer.layer_scale_2(layer.ffn(h))


class StreamCodec:
    """Streaming decode of one code frame for ``B`` slots (see the module doc).

    ``use_graph`` defaults to "on CUDA". ``compute_dtype`` (``torch.bfloat16``
    / ``torch.float16``) is honoured on CUDA only, as in the codec itself.
    """

    def __init__(self, tokenizer, B: int, *, compute_dtype: Optional[torch.dtype] = None,
                 use_graph: Optional[bool] = None, overlap: bool = False, warmup: int = 2):
        model = tokenizer
        p = next(model.parameters())
        dev = p.device
        self.model = model
        self.B = int(B)
        self.device = dev
        quantizer = getattr(model, "quantizer", None)
        if quantizer is None or not hasattr(quantizer, "decode_codes"):
            raise UnsupportedCodec("codec has no quantizer.decode_codes")
        self.n_vq = int(getattr(quantizer, "num_quantizers", 0) or 0)
        if self.n_vq <= 0:
            raise UnsupportedCodec("codec quantizer has no num_quantizers")
        self.autocast_dtype = compute_dtype if (dev.type == "cuda" and compute_dtype not in (None, torch.float32)) else None
        cache_dtype = self.autocast_dtype or p.dtype

        # The decoder is patch (upsample) modules and projected transformers;
        # a transformer's tokens per code frame = product of the patches before it.
        self._plan: list = []
        self.stages: List[_Stage] = []
        t = 1
        for mod in model.decoder:
            if hasattr(mod, "transformer") and hasattr(mod, "input_proj"):
                stage = _Stage(mod, t, self.B, dev, cache_dtype)
                self.stages.append(stage)
                self._plan.append(("stage", stage))
            elif hasattr(mod, "patch_size") and hasattr(mod, "decode"):
                t *= int(mod.patch_size)
                self._plan.append(("patch", mod))
            else:
                raise UnsupportedCodec(f"unknown decoder module {type(mod).__name__}")
        if not self.stages:
            raise UnsupportedCodec("codec decoder has no transformer")

        self.codes = torch.zeros(self.n_vq, self.B, 1, device=dev, dtype=torch.long)
        self.exec = torch.zeros(self.B, device=dev, dtype=torch.bool)
        self._lengths = torch.ones(self.B, device=dev, dtype=torch.long)
        self.out: Optional[torch.Tensor] = None
        self._mono: Optional[torch.Tensor] = None   # (F, B, SAMPLES_PER_FRAME), grown on demand
        self._host: Optional[torch.Tensor] = None   # pinned copy of ``_mono`` (overlap)
        self._last: Optional[CodecBlock] = None
        self.stream = torch.cuda.Stream(device=dev) if (overlap and dev.type == "cuda") else None

        self.graph = None
        if use_graph is None:
            use_graph = dev.type == "cuda"
        with torch.no_grad():
            if use_graph:
                # Warm-up and capture with every slot idle: offsets stay put and
                # the ring writes are overwritten by each slot's first real frame.
                s = torch.cuda.Stream()
                s.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(s):
                    for _ in range(max(1, warmup)):
                        self._frame()
                torch.cuda.current_stream().wait_stream(s)
                self.graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(self.graph):
                    self._frame()
            else:
                self._frame()   # allocates ``out`` with its real shape
        self.reset()

    # ── the frame ─────────────────────────────────────────────────────────────
    def _autocast(self):
        if self.autocast_dtype is None:
            return contextlib.nullcontext()
        return torch.autocast(device_type="cuda", dtype=self.autocast_dtype)

    def _frame(self) -> None:
        model = self.model
        x = model.quantizer.decode_codes(self.codes).float()      # (B, C, 1)
        with self._autocast():
            for kind, mod in self._plan:
                if kind == "patch":
                    x, _ = mod(x, self._lengths)
                else:
                    x = mod.forward(x, self.exec)
        audio, _ = model._restore_channels_from_codec(x, self._lengths)   # (B, ch, 3840) fp32
        if self.out is None:
            self.out = torch.zeros_like(audio)
        self.out.copy_(audio)

    def _on_stream(self):
        """Where every codec op runs: its own stream with ``overlap``, else the caller's."""
        return torch.cuda.stream(self.stream) if self.stream is not None else contextlib.nullcontext()

    # ── public ────────────────────────────────────────────────────────────────
    @torch.no_grad()
    def reset(self) -> None:
        """Empty every slot."""
        with self._on_stream():
            for st in self.stages:
                st.offset.zero_()

    @torch.no_grad()
    def reset_slots(self, slots: List[int]) -> None:
        """Start ``slots`` over (a finished stream's slot is reused)."""
        if not slots:
            return
        with self._on_stream():
            idx = torch.as_tensor(list(slots), device=self.device, dtype=torch.long)
            for st in self.stages:
                st.offset.index_fill_(0, idx, 0)

    @torch.no_grad()
    def step(self, codes: torch.Tensor, exec_mask: torch.Tensor) -> torch.Tensor:
        """One code frame for every slot.

        ``codes`` (n_vq, B) long, ``exec_mask`` (B,) bool — the slots that
        really have a frame now. Returns the static ``(B, channels, 3840)``
        output buffer, overwritten by the next call.
        """
        self.codes[:, :, 0].copy_(codes)
        self.exec.copy_(exec_mask)
        if self.graph is not None:
            self.graph.replay()
        else:
            self._frame()
        return self.out

    @torch.no_grad()
    def decode_block(self, codes: torch.Tensor, active: torch.Tensor) -> torch.Tensor:
        """``codes`` (F, B, n_vq) and ``active`` (F, B) bool, both on the codec's
        device → mono float32 audio ``(F, B, 3840)`` on the host. Slot ``b``'s
        frame ``f`` is decoded only where ``active[f, b]``; its audio is
        meaningless elsewhere."""
        return self.launch_block(codes, active).result()

    @torch.no_grad()
    def launch_block(self, codes: torch.Tensor, active: torch.Tensor) -> CodecBlock:
        """``decode_block`` that does not wait for the audio when ``overlap``
        is on. The previous block's audio is secured first (it shares buffers)."""
        if self._last is not None:
            self._last.result()
            self._last = None
        n = int(codes.shape[0])
        if self._mono is None or self._mono.shape[0] < n:
            self._mono = torch.zeros(n, self.B, self.out.shape[-1], device=self.device, dtype=torch.float32)
        if self.stream is None:
            for f in range(n):
                torch.mean(self.step(codes[f].transpose(0, 1), active[f]), dim=1, out=self._mono[f])
            # ``copy=True``: on a CPU codec ``.cpu()`` would hand out the buffer itself.
            return CodecBlock(None, self._mono[:n].to("cpu", copy=True))
        if self._host is None or self._host.shape[0] < n:
            self._host = torch.zeros(n, self.B, self._mono.shape[-1], dtype=torch.float32, pin_memory=True)
        inputs_ready = torch.cuda.Event()
        inputs_ready.record()                                  # codes / mask written on the caller's stream
        with self._on_stream():
            self.stream.wait_event(inputs_ready)
            codes.record_stream(self.stream)
            active.record_stream(self.stream)
            for f in range(n):
                torch.mean(self.step(codes[f].transpose(0, 1), active[f]), dim=1, out=self._mono[f])
            self._host[:n].copy_(self._mono[:n], non_blocking=True)
            done = torch.cuda.Event()
            done.record(self.stream)
        self._last = CodecBlock(done, self._host[:n])
        return self._last
