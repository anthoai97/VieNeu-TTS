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

Fast path (fp32, ``VIENEU_FASTPATH`` != 0; RTX 5060 Ti, 2026-10-24): a frame
was 2.6 ms for 1 slot and 4.5 ms for 16, ~45 kernels a layer — the codec's
RoPE alone recomputes its frequencies and cos/sin in ~25 kernels per layer,
and its 16-codebook quantizer runs 16 weight-normed 1x1 convs per frame. Now:

* the quantizer is one precomputed table per codebook (codebook → its
  ``out_proj`` → the shared ``output_proj``; all linear, so exact up to
  summation order) and a frame's 16 codes are one ``embedding_bag``;
* per stage, once per frame: the sliding-window mask as an additive bias,
  the scatter index and the rotary angles (the codec's own formula), applied
  to q and k of every layer as one complex multiply;
* per layer: the attention scale (1/8, exact) folded into ``in_proj``, both
  layer scales into the projection before them, residual adds into ``addmm``,
  attention as ``baddbmm`` + softmax + ``bmm`` — 15 kernels;
* one graph per slot bucket (1, 2, 4, ...): a step for slots ``[:n]`` only
  computes those, so one listener costs a 1-slot frame. ``stream.py`` hands
  out the lowest free slot, so the busy slots are a prefix.
"""
from __future__ import annotations

import contextlib
import math
import os
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

    def __init__(self, module, T: int, B: int, device, dtype, fast: bool = False):
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
        self.fast = bool(fast)
        if self.fast:
            self._prepare_fast(device)

    @torch.no_grad()
    def _prepare_fast(self, device) -> None:
        a0 = self.layers[0].self_attn
        rope = getattr(a0, "rope", None)
        if rope is None:
            raise UnsupportedCodec("decoder attention without rope")
        D = self.D
        # apply_rope's own frequencies, same expression.
        ds = torch.arange(D // 2, device=device, dtype=torch.float32)
        self.freqs = torch.exp(ds * (-math.log(rope.max_period) * 2 / D))
        self._tf = self._t.float()
        self._one = torch.ones(1, device=device)
        self.wqkv, self.wo, self.w1, self.w2 = [], [], [], []
        for layer in self.layers:
            a = layer.self_attn
            ffn = layer.ffn
            if (a.in_proj.bias is not None or a.out_proj.bias is not None
                    or not isinstance(ffn, torch.nn.Sequential) or len(ffn) != 3
                    or not isinstance(ffn[1], torch.nn.GELU) or ffn[0].bias is not None
                    or ffn[2].bias is not None):
                raise UnsupportedCodec("decoder layer layout")

            def scale(ls):
                s = getattr(ls, "scale", None)
                return torch.ones(self.E, device=device) if s is None else s.detach().float()

            w = a.in_proj.weight.detach().float().clone()
            w[:self.E] *= D ** -0.5                   # SDPA's default scale, exact (1/8)
            self.wqkv.append(w)
            self.wo.append(a.out_proj.weight.detach().float() * scale(layer.layer_scale_1).view(-1, 1))
            self.w1.append(ffn[0].weight.detach().float())
            self.w2.append(ffn[2].weight.detach().float() * scale(layer.layer_scale_2).view(-1, 1))
            if ffn[1].approximate != "none":
                raise UnsupportedCodec("tanh GELU")

    def forward_fast(self, x: torch.Tensor, exec_mask: torch.Tensor, n: int) -> torch.Tensor:
        """``forward`` for slots ``[:n]`` with this module's arithmetic (fp32)."""
        m = self.module
        H, D, T, R, E = self.H, self.D, self.T, self.R, self.E
        x = m.input_proj(x.transpose(1, 2))                         # (n, T, E)
        off = self.offset[:n]
        pos_q = off.view(-1, 1) + self._t                           # (n, T)
        end = pos_q[:, -1:]
        pos_k = end - torch.remainder(end - self._r, R)             # (n, R)
        delta = pos_q.unsqueeze(-1) - pos_k.unsqueeze(1)            # (n, T, R)
        ok = (pos_k >= 0).unsqueeze(1) & (delta >= 0) & (delta < self.ctx)
        bias = torch.zeros(n, 1, T, R, device=x.device).masked_fill_(~ok.unsqueeze(1), float("-inf"))
        bias = bias.expand(n, H, T, R).reshape(n * H, T, R)
        idx = torch.remainder(pos_q, R).view(n, 1, T, 1).expand(n, H, T, D)
        ts = off.float().view(-1, 1) + self._tf                     # (n, T)
        rot = torch.polar(self._one, self.freqs * ts.view(n, T, 1)).view(1, n, 1, T, D // 2)
        for i, layer in enumerate(self.layers):
            x = self._layer_fast(i, layer, x, bias, rot, idx, n)
        self.offset[:n].copy_(torch.where(exec_mask, off + T, off))
        return m.output_proj(x).transpose(1, 2)

    def _layer_fast(self, i: int, layer, x, bias, rot, idx, n: int) -> torch.Tensor:
        H, D, T, R, E = self.H, self.D, self.T, self.R, self.E
        qkv = F.linear(layer.norm1(x), self.wqkv[i]).view(n, T, 3, H, D)
        # q and k as complex pairs (re, im) = (x[2j], x[2j+1]), rotated together.
        qk = torch.view_as_complex(qkv[:, :, :2].permute(2, 0, 3, 1, 4).unflatten(-1, (D // 2, 2)))
        qk = torch.view_as_real(qk * rot).flatten(-2)               # (2, n, H, T, D)
        kc, vc = self.k[i, :n], self.v[i, :n]                       # (n, H, R, D)
        kc.scatter_(2, idx, qk[1])
        vc.scatter_(2, idx, qkv[:, :, 2].permute(0, 2, 1, 3))
        s = torch.baddbmm(bias, qk[0].reshape(n * H, T, D), kc.view(n * H, R, D).transpose(1, 2))
        o = torch.bmm(torch.softmax(s, dim=-1), vc.view(n * H, R, D))
        o = o.view(n, H, T, D).transpose(1, 2).reshape(n * T, E)
        x2 = torch.addmm(x.reshape(n * T, E), o, self.wo[i].t())
        f = F.gelu(F.linear(layer.norm2(x2), self.w1[i]))
        return torch.addmm(x2, f, self.w2[i].t()).view(n, T, E)

    @torch.no_grad()
    def forward_full(self, x: torch.Tensor, block: int = 512) -> torch.Tensor:
        """The whole sequence at once (non-streaming decode, positions from 0):
        ``x`` (B, C, Ts) → (B, C', Ts). Causal sliding-window attention in
        query blocks of ``block`` tokens; same arithmetic as ``forward_fast``."""
        m = self.module
        H, D, E, ctx = self.H, self.D, self.E, self.ctx
        x = m.input_proj(x.transpose(1, 2))                         # (B, Ts, E)
        B, Ts = x.shape[0], x.shape[1]
        t = torch.arange(Ts, device=x.device)
        rot = torch.polar(self._one, self.freqs * t.float().view(Ts, 1)).view(1, 1, 1, Ts, D // 2)
        spans = []
        for q0 in range(0, Ts, block):
            q1 = min(Ts, q0 + block)
            k0 = max(0, q0 - ctx + 1)
            delta = t[q0:q1].view(-1, 1) - t[k0:q1].view(1, -1)
            bias = torch.zeros(delta.shape, device=x.device).masked_fill_((delta < 0) | (delta >= ctx), float("-inf"))
            spans.append((q0, q1, k0, bias.unsqueeze(0)))
        for i, layer in enumerate(self.layers):
            qkv = F.linear(layer.norm1(x), self.wqkv[i]).view(B, Ts, 3, H, D)
            qk = torch.view_as_complex(qkv[:, :, :2].permute(2, 0, 3, 1, 4).unflatten(-1, (D // 2, 2)))
            qk = torch.view_as_real(qk * rot).flatten(-2)           # (2, B, H, Ts, D)
            q, k = qk[0].reshape(B * H, Ts, D), qk[1].reshape(B * H, Ts, D)
            v = qkv[:, :, 2].permute(0, 2, 1, 3).reshape(B * H, Ts, D)
            o = [torch.bmm(torch.softmax(torch.baddbmm(bias.expand(B * H, -1, -1), q[:, q0:q1],
                                                       k[:, k0:q1].transpose(1, 2)), dim=-1), v[:, k0:q1])
                 for q0, q1, k0, bias in spans]
            o = (o[0] if len(o) == 1 else torch.cat(o, dim=1)).view(B, H, Ts, D).transpose(1, 2).reshape(B * Ts, E)
            x2 = torch.addmm(x.reshape(B * Ts, E), o, self.wo[i].t())
            f = F.gelu(F.linear(layer.norm2(x2), self.w1[i]))
            x = torch.addmm(x2, f, self.w2[i].t()).view(B, Ts, E)
        return m.output_proj(x).transpose(1, 2)

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
                 use_graph: Optional[bool] = None, overlap: bool = False, warmup: int = 2,
                 fast: Optional[bool] = None):
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
        # The fast arithmetic is written for fp32 (the codec's own dtype); a
        # bf16/fp16 decoder keeps the module path under autocast.
        if fast is None:
            fast = os.environ.get("VIENEU_FASTPATH", "1") != "0"
        self.fast = bool(fast) and self.autocast_dtype is None and p.dtype == torch.float32

        # The decoder is patch (upsample) modules and projected transformers;
        # a transformer's tokens per code frame = product of the patches before it.
        def build(fast_stages: bool):
            plan, stages, t = [], [], 1
            for mod in model.decoder:
                if hasattr(mod, "transformer") and hasattr(mod, "input_proj"):
                    stage = _Stage(mod, t, self.B, dev, cache_dtype, fast=fast_stages)
                    stages.append(stage)
                    plan.append(("stage", stage))
                elif hasattr(mod, "patch_size") and hasattr(mod, "decode"):
                    t *= int(mod.patch_size)
                    plan.append(("patch", mod))
                else:
                    raise UnsupportedCodec(f"unknown decoder module {type(mod).__name__}")
            if not stages:
                raise UnsupportedCodec("codec decoder has no transformer")
            return plan, stages

        if self.fast:
            try:
                self._plan, self.stages = build(True)
                self._prepare_quantizer(quantizer)
            except UnsupportedCodec:
                self.fast = False
        if not self.fast:
            self._plan, self.stages = build(False)

        self.codes = torch.zeros(self.n_vq, self.B, 1, device=dev, dtype=torch.long)
        self.exec = torch.zeros(self.B, device=dev, dtype=torch.bool)
        self._lengths = torch.ones(self.B, device=dev, dtype=torch.long)
        self.out: Optional[torch.Tensor] = None
        self._mono: Optional[torch.Tensor] = None   # (F, B, SAMPLES_PER_FRAME), grown on demand
        self._host: Optional[torch.Tensor] = None   # pinned copy of ``_mono`` (overlap)
        self._last: Optional[CodecBlock] = None
        self.stream = torch.cuda.Stream(device=dev) if (overlap and dev.type == "cuda") else None
        # Slot buckets: a fast step over slots [:n] computes only those.
        self.buckets = [self.B]
        if self.fast:
            self.buckets = [1 << i for i in range(self.B.bit_length()) if (1 << i) < self.B] + [self.B]

        self.graph = None
        self.graphs = {}
        if use_graph is None:
            use_graph = dev.type == "cuda"
        with torch.no_grad():
            self._frame(self.B)   # allocates ``out`` with its real shape
            if use_graph:
                # Warm-up and capture with every slot idle: offsets stay put and
                # the ring writes are overwritten by each slot's first real frame.
                pool = torch.cuda.graph_pool_handle()
                for n in self.buckets:
                    s = torch.cuda.Stream()
                    s.wait_stream(torch.cuda.current_stream())
                    with torch.cuda.stream(s):
                        for _ in range(max(1, warmup) if not self.graphs else 1):
                            self._frame(n)
                    torch.cuda.current_stream().wait_stream(s)
                    g = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(g, pool=pool):
                        self._frame(n)
                    self.graphs[n] = g
                self.graph = self.graphs[self.B]
        self.reset()

    @torch.no_grad()
    def _prepare_quantizer(self, quantizer) -> None:
        """Per-codebook tables of the whole quantizer decode (fp32):
        ``decode_codes(c) == sum_i table[i][c_i] + const`` — codebook row →
        that codebook's ``out_proj`` → the shared ``output_proj``, all linear."""
        def conv_wb(conv):
            if not isinstance(conv, torch.nn.Conv1d) or tuple(conv.kernel_size) != (1,):
                raise UnsupportedCodec("quantizer projections are not 1x1 convs")
            w = conv.weight.detach().float()[:, :, 0]          # (out, in); weight norm applied
            b = conv.bias.detach().float() if conv.bias is not None else w.new_zeros(w.shape[0])
            return w, b

        qs = list(getattr(quantizer, "quantizers", []))[:self.n_vq]
        if len(qs) != self.n_vq or any(not hasattr(q, "codebook") for q in qs):
            raise UnsupportedCodec("quantizer layout")
        tables, bias = [], None
        for q in qs:
            t = q.codebook.weight.detach().float()             # (V, d)
            if not isinstance(q.out_proj, torch.nn.Identity):
                w, b = conv_wb(q.out_proj)
                t = t @ w.t()
                bias = b if bias is None else bias + b
            tables.append(t)
        tables = torch.stack(tables)                           # (n_vq, V, rvq)
        const = tables.new_zeros(tables.shape[-1]) if bias is None else bias
        out = getattr(quantizer, "output_proj", torch.nn.Identity())
        if not isinstance(out, torch.nn.Identity):
            wo, bo = conv_wb(out)
            tables = tables @ wo.t()
            const = const @ wo.t() + bo
        V = tables.shape[1]
        self._q_table = tables.reshape(-1, tables.shape[-1]).contiguous()
        self._q_const = const.contiguous()
        self._q_offsets = (torch.arange(self.n_vq, device=tables.device) * V).view(1, -1)

    # ── the frame ─────────────────────────────────────────────────────────────
    def _autocast(self):
        if self.autocast_dtype is None:
            return contextlib.nullcontext()
        return torch.autocast(device_type="cuda", dtype=self.autocast_dtype)

    def _frame(self, n: Optional[int] = None) -> None:
        """One code frame for slots ``[:n]`` (every slot on the module path)."""
        model = self.model
        if not self.fast:
            n = self.B
            x = model.quantizer.decode_codes(self.codes).float()      # (B, C, 1)
            with self._autocast():
                for kind, mod in self._plan:
                    if kind == "patch":
                        x, _ = mod(x, self._lengths)
                    else:
                        x = mod.forward(x, self.exec)
        else:
            n = self.B if n is None else n
            ids = self.codes[:, :n, 0].t() + self._q_offsets            # (n, n_vq)
            x = (F.embedding_bag(ids, self._q_table, mode="sum") + self._q_const).unsqueeze(-1)
            lengths, exec_mask = self._lengths[:n], self.exec[:n]
            for kind, mod in self._plan:
                if kind == "patch":
                    x, _ = mod(x, lengths)
                else:
                    x = mod.forward_fast(x, exec_mask, n)
        audio, _ = model._restore_channels_from_codec(x, self._lengths[:n])   # (n, ch, 3840) fp32
        if self.out is None:
            self.out = torch.zeros_like(audio)
        self.out[:n].copy_(audio)

    @torch.no_grad()
    def decode_full(self, codes: List[torch.Tensor]) -> List[torch.Tensor]:
        """Non-streaming decode of whole rows ``(T_i, n_vq)`` → mono float32
        audio ``(T_i * 3840,)`` on the host, every row in one pass. Rows are
        padded at the end (the decoder is causal: padding never reaches earlier
        samples). Fast stages only; no slot state is touched.

        Replaces the codec's own ``batch_decode`` for whole utterances: on an
        RTX 5060 Ti that took 24 ms for 55 frames (~3,700 eager ops, and its
        1x1 quantizer convs run in TF32 — ~3e-3 off fp32)."""
        if not self.fast:
            raise RuntimeError("decode_full needs the fast stages (fp32 codec, VIENEU_FASTPATH != 0)")
        lens = [int(c.shape[0]) for c in codes]
        B, T = len(codes), max(lens)
        ids = torch.nn.utils.rnn.pad_sequence([c.to(self.device, torch.long) for c in codes], batch_first=True)
        x = F.embedding_bag((ids + self._q_offsets.view(1, 1, -1)).view(B * T, self.n_vq), self._q_table, mode="sum")
        x = (x + self._q_const).view(B, T, -1).transpose(1, 2)            # (B, C, T)
        lengths = torch.full((B,), T, device=self.device, dtype=torch.long)
        for kind, mod in self._plan:
            if kind == "patch":
                x, lengths = mod(x, lengths)
            else:
                x = mod.forward_full(x)
        audio, _ = self.model._restore_channels_from_codec(x, lengths)       # (B, ch, T * 3840)
        mono = audio.mean(1).cpu()
        return [mono[b, :lens[b] * SAMPLES_PER_FRAME] for b in range(B)]

    def _on_stream(self):
        """Where every codec op runs: its own stream with ``overlap``, else the caller's."""
        return torch.cuda.stream(self.stream) if self.stream is not None else contextlib.nullcontext()

    def bucket(self, n_active: Optional[int]) -> int:
        """Smallest slot bucket covering slots ``[:n_active]`` (all of them for ``None``)."""
        if n_active is None:
            return self.B
        n = max(1, int(n_active))
        return next(b for b in self.buckets if b >= n)

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
    def step(self, codes: torch.Tensor, exec_mask: torch.Tensor, n_active: Optional[int] = None) -> torch.Tensor:
        """One code frame for every slot.

        ``codes`` (n_vq, B) long, ``exec_mask`` (B,) bool — the slots that
        really have a frame now. With ``n_active``, no slot from there on may
        be active: only slots ``[:bucket(n_active)]`` are computed (the rest
        of the output is stale). Returns the static ``(B, channels, 3840)``
        output buffer, overwritten by the next call.
        """
        self.codes[:, :, 0].copy_(codes)
        self.exec.copy_(exec_mask)
        n = self.bucket(n_active)
        if self.graphs:
            self.graphs[n].replay()
        else:
            self._frame(n)
        return self.out

    @torch.no_grad()
    def decode_block(self, codes: torch.Tensor, active: torch.Tensor,
                     n_active: Optional[int] = None) -> torch.Tensor:
        """``codes`` (F, B, n_vq) and ``active`` (F, B) bool, both on the codec's
        device → mono float32 audio ``(F, B, 3840)`` on the host. Slot ``b``'s
        frame ``f`` is decoded only where ``active[f, b]``; its audio is
        meaningless elsewhere."""
        return self.launch_block(codes, active, n_active).result()

    @torch.no_grad()
    def launch_block(self, codes: torch.Tensor, active: torch.Tensor,
                     n_active: Optional[int] = None) -> CodecBlock:
        """``decode_block`` that does not wait for the audio when ``overlap``
        is on. The previous block's audio is secured first (it shares buffers).
        ``n_active`` as in ``step``."""
        if self._last is not None:
            self._last.result()
            self._last = None
        n = int(codes.shape[0])
        nb = self.bucket(n_active)
        if self._mono is None or self._mono.shape[0] < n:
            self._mono = torch.zeros(n, self.B, self.out.shape[-1], device=self.device, dtype=torch.float32)
        if self.stream is None:
            for f in range(n):
                out = self.step(codes[f].transpose(0, 1), active[f], nb)
                torch.mean(out[:nb], dim=1, out=self._mono[f, :nb])
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
                out = self.step(codes[f].transpose(0, 1), active[f], nb)
                torch.mean(out[:nb], dim=1, out=self._mono[f, :nb])
            self._host[:n].copy_(self._mono[:n], non_blocking=True)
            done = torch.cuda.Event()
            done.record(self.stream)
        self._last = CodecBlock(done, self._host[:n])
        return self._last
