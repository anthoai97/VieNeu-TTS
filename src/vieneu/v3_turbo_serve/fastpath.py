"""
Low-launch rewrite of the v3 Turbo GPU hot path (same model, same maths).
=========================================================================

Why
---
On an RTX 5060 Ti (2026-10-24) a fused frame (``fused.FusedFrame``) took
5.8 ms whether it carried 1 row or 16: one CUDA graph node costs ~1.5 us and
the frame had ~2,300 of them. The arithmetic is tiny (the backbone's bf16
weights stream in ~0.5 ms), so the frame was bound by the number of kernels,
not by FLOPs or bandwidth. Measured per part, each as its own graph:

* backbone step 2.17 ms — HF RMSNorm is 7 kernels (x4 per layer), RoPE
  recomputes cos/sin and applies them in ~18 kernels, and SDPA's efficient
  kernel needs 55 us per layer for 3 query rows over a 1024-slot cache;
* acoustic decoder 1.81 ms (17 sequential steps of ~48 kernels);
* sampling 2.80 ms — ``sample_gpu_rows`` alone is 107 us a codebook (full
  sort + scatter + multinomial), and the repetition history did ~10 kernels a
  codebook on top.

The prompt side was worse: prefill and prompt embeddings ran eagerly (27 ms
for a 182-token prompt; one eager op costs 40-100 us on Windows/WDDM).

What
----
Nothing here changes the model; every rewrite below is exact in real
arithmetic and differs from the HF path only in bf16 rounding order.

* ``Prepared`` — the weights rearranged once per model: Q/K/V and gate/up
  projections concatenated; every pre-norm RMSNorm weight (and ``sqrt(D)``)
  folded into the projection that follows it, so a norm is ``||x||`` +
  ``hypot`` + one division after the GEMM; RoPE as a table of 64x64 rotation
  matrices (one ``bmm`` for all q and k heads); embeddings of all 17 prompt
  columns in one table (one ``embedding_bag`` per prompt or frame); the
  acoustic decoder's slot-position embeddings folded into its input tables
  and its final norm into the 16 audio heads and the text head.
* attention as ``baddbmm(out_dtype=fp32)`` + softmax + ``bmm`` — fp32
  scores like the fused kernels, 12 us instead of 55 us per decode layer.
* ``Sampler`` — top-k (or one sort when top-k varies per row), top-p on the
  renormalised top-k distribution (the single-path ``_sample_token`` rule),
  then an exponential race (``argmax(p / Exp(1))``), which samples exactly
  from the kept distribution (~5 us in a graph; ``multinomial`` alone was
  17 us) and, unlike ``multinomial``, cannot device-assert on the NaN rows of
  a continuous batch's idle slots. The repetition history is updated once
  per frame for all codebooks and turned into two multiplier tables, so a
  codebook's penalty + temperature is 4 kernels.
* ``Prefiller`` — the whole prompt (ids → embeddings → speaker anchor →
  12 layers → K/V + last hidden) as one CUDA graph per 32-token length
  bucket, shared by every frame object through a staging buffer.

``FastBatchFrame`` (static batch, ``engine.py``) and ``FastStreamFrame``
(continuous batch, ``stream.py``) put these together. CUDA only.
``VIENEU_FASTPATH=0`` goes back to ``fused.py`` / ``stream.StreamFrame``.
"""
from __future__ import annotations

import math
import os
import threading
from typing import Callable, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

from .._v3_turbo_engine.rep_history import DEFAULT_REP_WINDOW

_F32 = torch.float32


def enabled() -> bool:
    return os.environ.get("VIENEU_FASTPATH", "1") != "0"


# ── small ops ────────────────────────────────────────────────────────────────

def _mm32(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """``a @ b`` with a float32 result (half GEMM, fp32 output)."""
    return torch.mm(a, b) if a.dtype == _F32 else torch.mm(a, b, out_dtype=_F32)


def _bmm32(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.bmm(a, b) if a.dtype == _F32 else torch.bmm(a, b, out_dtype=_F32)


def _baddbmm32(bias: torch.Tensor, a: torch.Tensor, b: torch.Tensor, alpha: float) -> torch.Tensor:
    """``bias + alpha * (a @ b)`` in fp32. ``bias`` is expanded to the full
    output shape first: with ``out_dtype`` (torch 2.8) a broadcastable bias is
    NOT broadcast — the result silently takes the bias's batch size."""
    bias = bias.expand(a.shape[0], a.shape[1], b.shape[2])
    if a.dtype == _F32:
        return torch.baddbmm(bias, a, b, alpha=alpha)
    return torch.baddbmm(bias, a, b, alpha=alpha, out_dtype=_F32)


def _hyp(x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    """``sqrt(sum(x^2) + c^2)`` over the last dim, float32 ``(..., 1)``.

    ``rmsnorm(x) * w == x * (w * sqrt(D)) / _hyp(x, sqrt(D * eps))``.
    """
    return torch.hypot(torch.linalg.vector_norm(x, dim=-1, keepdim=True, dtype=_F32), c)


def _scaled(y: torch.Tensor, hyp: torch.Tensor) -> torch.Tensor:
    """``y / hyp`` computed in fp32, stored in ``y``'s dtype (one kernel)."""
    return torch.div(y, hyp, out=torch.empty_like(y))


def _norm_linear(x: torch.Tensor, w: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    """``F.linear(rmsnorm(x), W)`` for a ``w`` with the norm weight folded in (``Prepared``)."""
    return _scaled(F.linear(x, w), _hyp(x, c))


def _eps(module, dtype) -> float:
    e = getattr(module, "variance_epsilon", None)
    if e is None:
        e = getattr(module, "eps", None)
    return float(torch.finfo(dtype).eps if e is None else e)


# ── weights ──────────────────────────────────────────────────────────────────

class UnsupportedModel(RuntimeError):
    """The checkpoint has a layout this module was not written for."""


_PREP_LOCK = threading.Lock()


class Prepared:
    """The model's weights rearranged for the fast path; built once per model
    (``Prepared.of``) and shared read-only by every graph."""

    @classmethod
    def of(cls, model) -> "Prepared":
        with _PREP_LOCK:
            prep = model.__dict__.get("_vieneu_fastpath")
            if prep is None:
                prep = cls(model)
                model.__dict__["_vieneu_fastpath"] = prep
            return prep

    @torch.no_grad()
    def __init__(self, model):
        cfg = model.config
        bb = model.semantic_backbone
        p0 = next(model.parameters())
        dev, dt = p0.device, p0.dtype
        self.model = model
        self.device, self.dtype = dev, dt
        self.n_vq, self.V, self.H = int(cfg.n_vq), int(cfg.audio_vocab_size), int(cfg.hidden_size)
        self.n_text = int(cfg.text_vocab_size)
        self.sgs = int(cfg.speech_generation_start_token_id)
        self.eos = int(cfg.speech_generation_end_token_id)
        self.audio_pad = int(cfg.audio_pad_token_id)
        H = self.H
        f32 = lambda t: t.detach().to(device=dev, dtype=_F32)

        def fold(w, norm_w, D):
            return (f32(w) * (f32(norm_w) * math.sqrt(D)).view(1, -1)).to(dt).contiguous()

        def const(v):
            return torch.tensor(float(v), device=dev, dtype=_F32)

        # ── backbone (HF Qwen3) ──
        bcfg = bb.config
        if getattr(bcfg, "hidden_act", "silu") != "silu":
            raise UnsupportedModel(f"backbone activation {bcfg.hidden_act!r}")
        self.n_heads = int(bcfg.num_attention_heads)
        self.n_kv = int(bcfg.num_key_value_heads)
        self.hd = int(getattr(bcfg, "head_dim", None) or H // self.n_heads)
        self.g = self.n_heads // self.n_kv
        self.n_layers = len(bb.layers)
        self.max_pos = int(bcfg.max_position_embeddings)
        self.bb_wqkv, self.bb_wo, self.bb_wgu, self.bb_wd, self.bb_qkw = [], [], [], [], []
        self.bb_scaling = []
        for layer in bb.layers:
            a, mlp = layer.self_attn, layer.mlp
            for lin in (a.q_proj, a.k_proj, a.v_proj, a.o_proj, mlp.gate_proj, mlp.up_proj, mlp.down_proj):
                if lin.bias is not None:
                    raise UnsupportedModel("backbone projections with bias")
            self.bb_wqkv.append(fold(torch.cat([a.q_proj.weight, a.k_proj.weight, a.v_proj.weight]),
                                     layer.input_layernorm.weight, H))
            self.bb_wo.append(a.o_proj.weight.detach().contiguous())
            self.bb_wgu.append(fold(torch.cat([mlp.gate_proj.weight, mlp.up_proj.weight]),
                                    layer.post_attention_layernorm.weight, H))
            self.bb_wd.append(mlp.down_proj.weight.detach().contiguous())
            qw = f32(a.q_norm.weight) * math.sqrt(self.hd)
            kw = f32(a.k_norm.weight) * math.sqrt(self.hd)
            self.bb_qkw.append(torch.cat([qw.expand(self.n_heads, -1), kw.expand(self.n_kv, -1)]).to(dt).contiguous())
            self.bb_scaling.append(float(a.scaling))
        l0 = bb.layers[0]
        self.bb_c_in = const(math.sqrt(H * _eps(l0.input_layernorm, dt)))
        self.bb_c_post = const(math.sqrt(H * _eps(l0.post_attention_layernorm, dt)))
        self.bb_c_hd = const(math.sqrt(self.hd * _eps(l0.self_attn.q_norm, dt)))
        self.bb_final_w = (f32(bb.norm.weight) * math.sqrt(H)).to(dt)
        self.bb_c_final = const(math.sqrt(H * _eps(bb.norm, dt)))

        # RoPE: y = x @ R[pos], R built from HF's own cos/sin (rounded to the
        # model dtype, as HF does) in the rotate_half layout.
        rot = bb.rotary_emb
        inv_freq = f32(rot.inv_freq)
        att_scale = float(getattr(rot, "attention_scaling", 1.0))
        self.n_rope = 2 * self.max_pos
        t = torch.arange(self.n_rope, device=dev, dtype=_F32)
        fr = torch.outer(t, inv_freq)
        emb = torch.cat([fr, fr], dim=-1)
        cos, sin = (emb.cos() * att_scale).to(dt), (emb.sin() * att_scale).to(dt)
        half = self.hd // 2
        if inv_freq.numel() != half:
            raise UnsupportedModel("partial rotary embedding")
        R = torch.zeros(self.n_rope, self.hd, self.hd, device=dev, dtype=dt)
        i = torch.arange(half, device=dev)
        R[:, i, i] = cos[:, :half]
        R[:, i + half, i] = -sin[:, :half]
        R[:, i + half, i + half] = cos[:, half:]
        R[:, i, i + half] = sin[:, half:]
        self.rope = R

        # ── embeddings: one table for the text column and the 16 code columns
        # (each code table gets a zero row at ``audio_pad``) ──
        if self.audio_pad != self.V:
            raise UnsupportedModel("audio pad id is not the last code + 1")
        parts = [model.text_embeddings.weight.detach()]
        zero = torch.zeros(1, H, device=dev, dtype=parts[0].dtype)
        for emb_ch in model.audio_embeddings:
            parts += [emb_ch.weight.detach(), zero]
        self.emb_table = torch.cat(parts).to(dt).contiguous()
        self.emb_offsets = torch.tensor([0] + [self.n_text + ch * (self.V + 1) for ch in range(self.n_vq)],
                                        dtype=torch.long)
        self.code_offsets = self.emb_offsets[1:].to(dev)
        self.xvec = model.xvec_proj
        self.spk_dim = int(getattr(cfg, "speaker_embedding_dim", 0) or 0) if self.xvec is not None else 0

        # ── acoustic decoder ──
        dec = model.acoustic_decoder
        self.ac = []
        for layer in dec.layers:
            a = layer.attn
            nh, hd = int(a.num_heads), int(a.head_dim)
            # Attention scale 1/sqrt(hd) folded into q: q_norm.w * sqrt(hd) / sqrt(hd).
            qw = f32(a.q_norm.weight)
            kw = f32(a.k_norm.weight) * math.sqrt(hd)
            self.ac.append(dict(
                nh=nh, hd=hd,
                wqkv=fold(a.qkv.weight, layer.norm1.weight, H),
                qkw=torch.stack([qw.expand(nh, -1), kw.expand(nh, -1)]).to(dt).contiguous(),   # (2, nh, hd)
                wo=a.o_proj.weight.detach().contiguous(),
                wgu=fold(torch.cat([layer.ff_gate.weight, layer.ff_up.weight]), layer.norm2.weight, H),
                wd=layer.ff_down.weight.detach().contiguous(),
                inter=int(layer.ff_gate.weight.shape[0]),
                c_in=const(math.sqrt(H * _eps(layer.norm1, dt))),
                c_post=const(math.sqrt(H * _eps(layer.norm2, dt))),
                c_qk=const(math.sqrt(hd * _eps(a.q_norm, dt))),
            ))
        if any(abs(_eps(l.attn.q_norm, dt) - _eps(l.attn.k_norm, dt)) > 0 for l in dec.layers):
            raise UnsupportedModel("acoustic q_norm / k_norm with different eps")
        fw = f32(dec.norm.weight) * math.sqrt(H)
        self.ac_heads = torch.stack([f32(h.weight) * fw for h in model.audio_lm_heads]).to(dt).contiguous()
        self.ac_c_final = const(math.sqrt(H * _eps(dec.norm, dt)))
        self.text_head = (f32(model.text_lm_head.weight) * fw).to(dt).contiguous()
        pos = dec.slot_pos_emb.weight.detach().to(dt)
        self.ac_pos0 = pos[0].contiguous()
        sgs_emb = model.text_embeddings.weight[self.sgs].detach().to(dt)
        self.ac_x1 = (sgs_emb + pos[1]).contiguous()
        # Step ch (1..n_vq-1) feeds code ch-1 through audio_embeddings[ch-1] at slot ch+1.
        self.ac_in = torch.stack([model.audio_embeddings[ch - 1].weight.detach().to(dt) + pos[ch + 1]
                                  for ch in range(1, self.n_vq)]).contiguous()          # (n_vq-1, V, H)
        self.sgs_emb = sgs_emb
        # Set after the first frame graph is captured: later captures need one
        # warm-up run instead of three (kernels and handles are initialised).
        self.warm = False

    def prompt_ids(self, prompt_2d: torch.Tensor) -> torch.Tensor:
        """``(T, n_vq+1)`` prompt (``build_prompt_2d``) → row ids into ``emb_table``."""
        return prompt_2d.to(torch.long) + self.emb_offsets.view(1, -1)


# ── backbone ─────────────────────────────────────────────────────────────────

class FastBackbone:
    """Qwen3 decode over a preallocated ring KV cache for ``B`` rows.

    Same contract as ``fused.StaticBackbone`` (rows right-aligned to one
    shared write index ``cur``; per-row rotary positions), with the arithmetic
    of this module. ``step`` is a fixed sequence of device ops.
    """

    def __init__(self, prep: Prepared, B: int, max_len: int):
        self.p = prep
        self.B, self.max_len = int(B), int(max_len)
        dev, dt = prep.device, prep.dtype
        L, nkv, hd = prep.n_layers, prep.n_kv, prep.hd
        self.k = torch.zeros(L, self.B, nkv, self.max_len, hd, device=dev, dtype=dt)
        self.v = torch.zeros_like(self.k)
        # fp32 additive bias per (row, kv head, query of the group): 0 = may attend.
        self.bias = torch.full((self.B, nkv, prep.g, self.max_len), float("-inf"), device=dev, dtype=_F32)
        self.cur = torch.zeros((), device=dev, dtype=torch.long)
        self.pos = torch.zeros(self.B, device=dev, dtype=torch.long)

    def rows(self, n: int) -> "FastBackbone":
        """Slots ``[:n]`` as a backbone of their own: views of this cache, same write index."""
        v = object.__new__(FastBackbone)
        v.p, v.B, v.max_len, v.cur = self.p, int(n), self.max_len, self.cur
        v.k, v.v, v.bias, v.pos = self.k[:, :n], self.v[:, :n], self.bias[:n], self.pos[:n]
        return v

    @torch.no_grad()
    def reset(self) -> None:
        self.bias.fill_(float("-inf"))
        self.cur.zero_()
        self.pos.zero_()

    @torch.no_grad()
    def load_row(self, b: int, cur: int, T: int, k_rows: torch.Tensor, v_rows: torch.Tensor) -> None:
        """Slot ``b`` takes a prefilled prompt of ``T`` tokens: ``k_rows`` /
        ``v_rows`` are ``(L, n_kv, T, hd)``, written right-aligned to the
        write index ``cur`` (host mirror of ``self.cur``). Host-side."""
        if T + 1 > self.max_len:
            raise ValueError(f"prompt of {T} tokens does not fit a cache of {self.max_len}")
        idx = torch.remainder(torch.arange(cur - T, cur), self.max_len).to(self.k.device, non_blocking=True)
        self.k[:, b].index_copy_(2, idx, k_rows)
        self.v[:, b].index_copy_(2, idx, v_rows)
        self.bias[b].fill_(float("-inf"))
        self.bias[b].index_fill_(-1, idx, 0.0)
        self.pos[b] = T

    def step(self, x: torch.Tensor) -> torch.Tensor:
        """``x`` (B, H) input embedding → final hidden (B, H). Advances the cache."""
        p = self.p
        B, nh, nkv, hd, g, L = self.B, p.n_heads, p.n_kv, p.hd, p.g, self.max_len
        widx = torch.remainder(self.cur, L).view(1)
        self.bias.view(-1, L).index_fill_(1, widx, 0.0)          # the new token sees itself
        R = p.rope.index_select(0, self.pos.clamp(max=p.n_rope - 1))   # (B, hd, hd)
        bias = self.bias.view(B * nkv, g, L)
        for i in range(p.n_layers):
            qkv = _norm_linear(x, p.bb_wqkv[i], p.bb_c_in)                         # (B, (nh+2nkv)hd)
            qk = qkv[:, :(nh + nkv) * hd].view(B, nh + nkv, hd)
            qkn = _scaled(qk * p.bb_qkw[i], _hyp(qk, p.bb_c_hd))
            q = torch.bmm(qkn[:, :nh], R)                                          # (B, nh, hd)
            k = torch.bmm(qkn[:, nh:], R)                                          # (B, nkv, hd)
            v = qkv[:, (nh + nkv) * hd:].view(B, nkv, 1, hd)
            kc, vc = self.k[i], self.v[i]
            kc.index_copy_(2, widx, k.view(B, nkv, 1, hd))
            vc.index_copy_(2, widx, v)
            s = _baddbmm32(bias, q.view(B * nkv, g, hd), kc.view(B * nkv, L, hd).transpose(1, 2),
                           p.bb_scaling[i])
            o = torch.bmm(torch.softmax(s, dim=-1).to(x.dtype), vc.view(B * nkv, L, hd))
            x = torch.addmm(x, o.view(B, nh * hd), p.bb_wo[i].t())
            gu = _norm_linear(x, p.bb_wgu[i], p.bb_c_post)
            I = gu.shape[-1] // 2
            x = torch.addmm(x, F.silu(gu[:, :I]) * gu[:, I:], p.bb_wd[i].t())
        self.cur += 1
        self.pos += 1
        return _scaled(x * p.bb_final_w, _hyp(x, p.bb_c_final))


# ── prompt prefill ───────────────────────────────────────────────────────────

def prefill_bucket(T: int, step: int = 32) -> int:
    return max(step, int(math.ceil(T / step)) * step)


class Prefiller:
    """Prompt ids → backbone K/V + last hidden, one CUDA graph per length bucket.

    One per model (``Prefiller.of``); every frame object copies the result out
    of the staging buffers (``k``/``v``/``h_last``/``spk_h``) under ``lock``.
    A prompt of ``T`` tokens runs in bucket ``Tb >= T``, left-padded; padded
    keys are masked and only the last ``T`` positions are read back.
    """

    STEP = 32

    @classmethod
    def of(cls, prep: Prepared) -> "Prefiller":
        with _PREP_LOCK:
            pf = prep.__dict__.get("_prefiller")
            if pf is None:
                pf = prep.__dict__["_prefiller"] = cls(prep)
            return pf

    def __init__(self, prep: Prepared):
        self.p = prep
        dev, dt = prep.device, prep.dtype
        self.T_max = prefill_bucket(prep.max_pos, self.STEP)
        self.lock = threading.RLock()
        n_col = prep.n_vq + 1
        self.ids = torch.zeros(self.T_max, n_col, device=dev, dtype=torch.long)
        self._host_ids = torch.zeros(self.T_max, n_col, dtype=torch.long,
                                     pin_memory=dev.type == "cuda")
        self._host_free: Optional["torch.cuda.Event"] = None
        self.n_pad = torch.zeros((), device=dev, dtype=torch.long)
        self.spk = torch.zeros(1, max(prep.spk_dim, 1), device=dev, dtype=dt)
        self.has_spk = torch.zeros((), device=dev, dtype=dt)
        self.k = torch.zeros(prep.n_layers, prep.n_kv, self.T_max, prep.hd, device=dev, dtype=dt)
        self.v = torch.zeros_like(self.k)
        self.h_last = torch.zeros(prep.H, device=dev, dtype=dt)
        self.spk_h = torch.zeros(prep.H, device=dev, dtype=dt)
        self._ar = torch.arange(self.T_max, device=dev)
        self.graphs = {}
        self.use_graph = dev.type == "cuda"
        self._pool = torch.cuda.graph_pool_handle() if self.use_graph else None

    def _body(self, Tb: int) -> None:
        p = self.p
        nh, nkv, hd, g = p.n_heads, p.n_kv, p.hd, p.g
        ids = self.ids[:Tb]
        x = F.embedding_bag(ids, p.emb_table, mode="sum")                          # (Tb, H)
        if p.xvec is not None:
            spk_h = p.xvec(self.spk) * self.has_spk                                # (1, H)
        else:
            spk_h = torch.zeros(1, p.H, device=x.device, dtype=x.dtype)
        self.spk_h.copy_(spk_h[0])
        x = x + spk_h
        ar = self._ar[:Tb]
        pos = (ar - self.n_pad).clamp(min=0)
        R = p.rope.index_select(0, pos)                                            # (Tb, hd, hd)
        allowed = ((ar.view(1, -1) >= self.n_pad) & (ar.view(1, -1) <= ar.view(-1, 1))) | (ar.view(1, -1) == ar.view(-1, 1))
        bias = torch.zeros(Tb, Tb, device=x.device, dtype=_F32).masked_fill_(~allowed, float("-inf"))
        bias = bias.repeat(g, 1).view(1, g * Tb, Tb)
        for i in range(p.n_layers):
            qkv = _norm_linear(x, p.bb_wqkv[i], p.bb_c_in)
            qk = qkv[:, :(nh + nkv) * hd].view(Tb, nh + nkv, hd)
            qkn = _scaled(qk * p.bb_qkw[i], _hyp(qk, p.bb_c_hd))
            qk = torch.bmm(qkn, R)                                                # (Tb, nh+nkv, hd)
            k = qk[:, nh:]
            v = qkv[:, (nh + nkv) * hd:].view(Tb, nkv, hd)
            self.k[i, :, :Tb].copy_(k.transpose(0, 1))
            self.v[i, :, :Tb].copy_(v.transpose(0, 1))
            # q heads h = j*g + r share kv head j: (nkv, g*Tb, hd), rows (r, t).
            q = qk[:, :nh].reshape(Tb, nkv, g, hd).permute(1, 2, 0, 3).reshape(nkv, g * Tb, hd)
            s = _baddbmm32(bias, q, self.k[i, :, :Tb].transpose(1, 2), p.bb_scaling[i])
            o = torch.bmm(torch.softmax(s, dim=-1).to(x.dtype), self.v[i, :, :Tb])  # (nkv, g*Tb, hd)
            o = o.view(nkv, g, Tb, hd).permute(2, 0, 1, 3).reshape(Tb, nh * hd)
            x = torch.addmm(x, o, p.bb_wo[i].t())
            gu = _norm_linear(x, p.bb_wgu[i], p.bb_c_post)
            I = gu.shape[-1] // 2
            x = torch.addmm(x, F.silu(gu[:, :I]) * gu[:, I:], p.bb_wd[i].t())
        last = x[-1:]
        self.h_last.copy_(_scaled(last * p.bb_final_w, _hyp(last, p.bb_c_final))[0])

    def _graph(self, Tb: int):
        g = self.graphs.get(Tb)
        if g is None:
            s = torch.cuda.Stream(device=self.p.device)
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s), torch.no_grad():
                for _ in range(1 if self.graphs else 2):
                    self._body(Tb)
            torch.cuda.current_stream().wait_stream(s)
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g, pool=self._pool, capture_error_mode="thread_local"), torch.no_grad():
                self._body(Tb)
            self.graphs[Tb] = g
        return g

    @torch.no_grad()
    def warm(self, max_tokens: int = 512) -> int:
        """Capture every bucket up to ``max_tokens``; returns how many were new."""
        n = 0
        with self.lock:
            for Tb in range(self.STEP, min(max_tokens, self.T_max) + 1, self.STEP):
                if Tb not in self.graphs and self.use_graph:
                    self._graph(Tb)
                    n += 1
        return n

    @torch.no_grad()
    def run(self, ids: torch.Tensor, spk: Optional[torch.Tensor]) -> Tuple[int, int]:
        """Prefill one prompt (``ids`` from ``Prepared.prompt_ids``, host
        ``(T, n_vq+1)``). Call with ``lock`` held and read the staging buffers
        before releasing it. Returns ``(T, Tb)``: the prompt's K/V are
        ``k[:, :, Tb-T:Tb]``."""
        T = int(ids.shape[0])
        if T == 0 or T > self.T_max:
            raise ValueError(f"prompt of {T} tokens (max {self.T_max})")
        Tb = prefill_bucket(T, self.STEP)
        if self._host_free is not None:
            self._host_free.synchronize()      # the previous upload has left the pinned buffer
        host = self._host_ids
        host[:Tb - T] = 0
        host[Tb - T:Tb].copy_(ids)
        self.ids[:Tb].copy_(host[:Tb], non_blocking=True)
        if self.use_graph:
            self._host_free = torch.cuda.Event()
            self._host_free.record()
        self.n_pad.fill_(Tb - T)
        if self.p.xvec is not None:
            if spk is None:
                self.has_spk.zero_()
            else:
                self.spk.copy_(spk.view(1, -1).to(self.spk.dtype), non_blocking=True)
                self.has_spk.fill_(1)
        if self.use_graph:
            self._graph(Tb).replay()
        else:
            self._body(Tb)
        return T, Tb


# ── sampling ─────────────────────────────────────────────────────────────────

class RowSampling:
    """Per-row sampling settings as device tensors (no recapture when they change).

    ``inv_t`` = 1/temperature (a huge value for greedy rows, which makes the
    race below pick the arg-max); ``top_p`` (``inf`` = off); ``k_idx`` =
    ``top_k - 1`` (``V - 1`` = off; only read when top-k varies per row);
    ``pen``/``inv_pen`` = repetition penalty and its inverse.
    """

    def __init__(self, B: int, V: int, device):
        self.B, self.V = B, V
        self.inv_t = torch.ones(B, 1, device=device)
        self.top_p = torch.full((B, 1), float("inf"), device=device)
        self.k_idx = torch.full((B, 1), V - 1, device=device, dtype=torch.long)
        self.pen = torch.ones(B, 1, 1, device=device)
        self.inv_pen = torch.ones(B, 1, 1, device=device)

    def rows(self, n: int) -> "RowSampling":
        v = object.__new__(RowSampling)
        v.B, v.V = int(n), self.V
        for name in ("inv_t", "top_p", "k_idx", "pen", "inv_pen"):
            setattr(v, name, getattr(self, name)[:n])
        return v

    @staticmethod
    def values(temperature: float, top_k: int, top_p: float, repetition_penalty: float, V: int):
        t = float(temperature)
        inv_t = 1.0 / t if t > 0 else 1e6
        k = int(top_k)
        k_idx = (min(k, V) - 1) if k > 0 else V - 1
        tp = float(top_p)
        tp = tp if 0.0 < tp < 1.0 else float("inf")
        pen = float(repetition_penalty)
        return inv_t, k_idx, tp, pen

    @torch.no_grad()
    def set_row(self, b: int, temperature: float, top_k: int, top_p: float, repetition_penalty: float) -> None:
        inv_t, k_idx, tp, pen = self.values(temperature, top_k, top_p, repetition_penalty, self.V)
        self.inv_t[b] = inv_t
        self.k_idx[b] = k_idx
        self.top_p[b] = tp
        self.pen[b] = pen
        self.inv_pen[b] = 1.0 / pen

    @torch.no_grad()
    def set_all(self, temperature: float, top_k: int, top_p: float, repetition_penalty: float) -> None:
        inv_t, k_idx, tp, pen = self.values(temperature, top_k, top_p, repetition_penalty, self.V)
        self.inv_t.fill_(inv_t)
        self.k_idx.fill_(k_idx)
        self.top_p.fill_(tp)
        self.pen.fill_(pen)
        self.inv_pen.fill_(1.0 / pen)


class FastRep:
    """Sliding-window repetition history, updated once per frame for all
    codebooks (same rule as ``fused.GpuRepHistory`` / ``RepetitionHistory``),
    plus the per-frame multiplier tables the sampler applies:
    ``scaled = logits * where(logits < 0, neg, pos)`` with
    ``pos = inv_t / penalty`` and ``neg = inv_t * penalty`` on seen codes."""

    def __init__(self, B: int, n_vq: int, V: int, window: int, device):
        self.window = int(window)
        self.counts = torch.zeros(B, n_vq, V, dtype=torch.int32, device=device)
        self.ring = torch.zeros(B, n_vq, max(self.window, 1), dtype=torch.long, device=device)
        self.frame = torch.zeros((), dtype=torch.long, device=device)
        self.start = torch.zeros(B, dtype=torch.long, device=device)
        self._ones = torch.ones(B, n_vq, 1, dtype=torch.int32, device=device)
        self.pos = torch.ones(B, n_vq, V, device=device)
        self.neg = torch.ones(B, n_vq, V, device=device)

    def rows(self, n: int) -> "FastRep":
        """Rows ``[:n]`` (views), sharing the frame counter."""
        v = object.__new__(FastRep)
        v.window, v.frame = self.window, self.frame
        for name in ("counts", "ring", "start", "_ones", "pos", "neg"):
            setattr(v, name, getattr(self, name)[:n])
        return v

    @torch.no_grad()
    def reset(self) -> None:
        self.counts.zero_()
        self.ring.zero_()
        self.frame.zero_()
        self.start.zero_()

    @torch.no_grad()
    def reset_rows(self, rows: torch.Tensor) -> None:
        self.counts[rows] = 0
        self.ring[rows] = 0
        self.start[rows] = self.frame

    def add_frame(self, codes: torch.Tensor) -> None:
        """Record one frame's ``codes`` (B, n_vq) and advance."""
        c = codes.unsqueeze(-1)
        if self.window > 0:
            slot = torch.remainder(self.frame, self.window).view(1)
            old = self.ring.index_select(2, slot)                                 # (B, n_vq, 1)
            evict = ((self.frame - self.start) >= self.window).to(torch.int32).view(-1, 1, 1)
            self.counts.scatter_add_(2, old, -(self._ones * evict))
            self.ring.index_copy_(2, slot, c)
        self.counts.scatter_add_(2, c, self._ones)
        self.frame += 1

    def tables(self, samp: RowSampling) -> None:
        """Refresh ``pos``/``neg`` for the next frame from the history."""
        seen = self.counts > 0
        it = samp.inv_t.view(-1, 1, 1)
        torch.where(seen, it * samp.inv_pen, it, out=self.pos)
        torch.where(seen, it * samp.pen, it, out=self.neg)


class Sampler:
    """Penalty + temperature + top-k + top-p + draw for one codebook, B rows.

    ``top_k`` given (the same for every row) → ``torch.topk``; ``None`` →
    one sort and a per-row cut at ``samp.k_idx``. Top-p is applied to the
    renormalised top-k distribution, as ``_sample_token`` does. The draw is an
    exponential race over the kept probabilities: exact, and an arg-max cannot
    fail on NaN rows (dead slots of a continuous batch).
    """

    def __init__(self, B: int, V: int, top_k: Optional[int], device):
        self.V = V
        self.K = None if top_k is None else (min(int(top_k), V) if int(top_k) > 0 else V)
        n = self.K or V
        self.e = torch.empty(B, n, device=device)

    def rows(self, n: int) -> "Sampler":
        v = object.__new__(Sampler)
        v.V, v.K, v.e = self.V, self.K, self.e[:n]
        return v

    def __call__(self, raw: torch.Tensor, hyp: torch.Tensor, ch: int, samp: RowSampling,
                 rep: Optional[FastRep]) -> torch.Tensor:
        """``raw`` (B, V) fp32 head output before the final 1/hyp scale."""
        if rep is not None:
            f = torch.where(raw < 0, rep.neg[:, ch], rep.pos[:, ch])
            scaled = (raw * f).div_(hyp)
        else:
            scaled = raw * (samp.inv_t / hyp)
        if self.K is not None and self.K < self.V:
            vals, idx = torch.topk(scaled, self.K, dim=-1)
        else:
            vals, idx = torch.sort(scaled, dim=-1, descending=True)
            if self.K is None:
                kth = vals.gather(1, samp.k_idx)
                vals = torch.where(vals >= kth, vals, float("-inf"))
        prob = torch.softmax(vals, dim=-1)
        keep = (prob.cumsum(dim=-1) - prob) < samp.top_p
        score = torch.where(keep, prob, 0.0).div_(self.e.exponential_())
        return idx.gather(1, score.argmax(dim=-1, keepdim=True)).squeeze(1)


# ── acoustic frame ───────────────────────────────────────────────────────────

class AcousticFrame:
    """One audio frame for B rows: acoustic decoder (17 slots), the 16
    codebook draws and the EOS flag, as a fixed sequence of device ops."""

    def __init__(self, prep: Prepared, B: int, sampler: Sampler):
        self.p = prep
        self.B = B
        self.sampler = sampler
        dev, dt = prep.device, prep.dtype
        S = prep.n_vq + 1
        self.k = [torch.zeros(B, a["nh"], S, a["hd"], device=dev, dtype=dt) for a in prep.ac]
        self.v = [torch.zeros_like(k) for k in self.k]
        self.xin = torch.zeros(B, 2, prep.H, device=dev, dtype=dt)
        self.xin[:, 1] = prep.ac_x1                                # slot 1 is constant
        self.bias2 = torch.tensor([[0.0, float("-inf")], [0.0, 0.0]], device=dev).view(1, 2, 2)
        self.codes = torch.zeros(B, prep.n_vq, device=dev, dtype=torch.long)

    def rows(self, n: int) -> "AcousticFrame":
        v = object.__new__(AcousticFrame)
        v.p, v.B, v.bias2, v.sampler = self.p, int(n), self.bias2, self.sampler.rows(n)
        v.k, v.v = [k[:n] for k in self.k], [x[:n] for x in self.v]
        v.xin, v.codes = self.xin[:n], self.codes[:n]
        return v

    def _layer(self, i: int, x: torch.Tensor, p0: int) -> torch.Tensor:
        """``x`` (B, n, H) for slots ``p0..p0+n-1`` → residual stream out."""
        a = self.p.ac[i]
        B, n, H = x.shape
        nh, hd = a["nh"], a["hd"]
        p1 = p0 + n
        x2 = x.reshape(B * n, H)
        qkv = _norm_linear(x2, a["wqkv"], a["c_in"])                              # (B*n, 3H)
        qk = qkv[:, :2 * H].view(B, n, 2, nh, hd)
        # Normalised q/k laid out (2, B, nh, n, hd) so both read back contiguous.
        buf = torch.empty(2, B, nh, n, hd, device=x.device, dtype=x.dtype)
        torch.div(qk * a["qkw"], _hyp(qk, a["c_qk"]), out=buf.permute(1, 3, 0, 2, 4))
        kc, vc = self.k[i], self.v[i]
        kc[:, :, p0:p1].copy_(buf[1])
        vc[:, :, p0:p1].copy_(qkv[:, 2 * H:].view(B, n, nh, hd).transpose(1, 2))
        q = buf[0].view(B * nh, n, hd)
        kt = kc[:, :, :p1].reshape(B * nh, p1, hd).transpose(1, 2)
        if n == 1:
            s = _bmm32(q, kt)
        else:
            s = _baddbmm32(self.bias2[:, :n, :p1], q, kt, 1.0)
        o = torch.bmm(torch.softmax(s, dim=-1).to(x.dtype), vc[:, :, :p1].reshape(B * nh, p1, hd))
        o = o.view(B, nh, n, hd).transpose(1, 2).reshape(B * n, H) if n > 1 else o.view(B, H)
        x2 = torch.addmm(x2, o, a["wo"].t())
        gu = _norm_linear(x2, a["wgu"], a["c_post"])
        I = a["inter"]
        x2 = torch.addmm(x2, F.silu(gu[:, :I]) * gu[:, I:], a["wd"].t())
        return x2.view(B, n, H)

    def __call__(self, h: torch.Tensor, samp: RowSampling, rep: Optional[FastRep]):
        """``h`` (B, H) backbone hidden → ``(codes (B, n_vq), is_eos (B,))``.
        ``rep`` must already hold this frame's tables (``FastRep.tables``)."""
        p = self.p
        B, H = self.B, p.H
        torch.add(h, p.ac_pos0, out=self.xin[:, 0])
        x = self.xin
        for i in range(len(p.ac)):
            x = self._layer(i, x, 0)
        # EOS: arg-max of the text head on slot 0 (a positive per-row norm
        # scale cannot change an arg-max, so it is skipped).
        is_eos = torch.mm(x[:, 0], p.text_head.t()).argmax(dim=-1) == p.eos
        last = x[:, 1]
        codes = self.codes
        for ch in range(p.n_vq):
            raw = _mm32(last, p.ac_heads[ch].t())                                   # (B, V) fp32
            code = self.sampler(raw, _hyp(last, p.ac_c_final), ch, samp, rep)
            codes[:, ch] = code
            if ch + 1 < p.n_vq:
                x = p.ac_in[ch].index_select(0, code).view(B, 1, H)
                for i in range(len(p.ac)):
                    x = self._layer(i, x, ch + 2)
                last = x[:, 0]
        return codes, is_eos


# ── frames ───────────────────────────────────────────────────────────────────

def _capture(body: Callable[[], None], device, warmup: int = 3, pool=None) -> "torch.cuda.CUDAGraph":
    s = torch.cuda.Stream(device=device)
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s), torch.no_grad():
        for _ in range(warmup):
            body()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g, pool=pool, capture_error_mode="thread_local"), torch.no_grad():
        body()
    return g


class _FrameBase:
    """State shared by both frame kinds: backbone, acoustic frame, history,
    per-row sampling, and the frame's feedback embedding."""

    def __init__(self, prep: Prepared, B: int, max_len: int, *, top_k_static: Optional[int],
                 use_rep: bool, repetition_window: int):
        self.p = prep
        self.B = int(B)
        dev, dt = prep.device, prep.dtype
        self.bb = FastBackbone(prep, self.B, max_len)
        self.samp = RowSampling(self.B, prep.V, dev)
        self.rep = FastRep(self.B, prep.n_vq, prep.V, repetition_window, dev) if use_rep else None
        self.ac = AcousticFrame(prep, self.B, Sampler(self.B, prep.V, top_k_static, dev))
        self.h = torch.zeros(self.B, prep.H, device=dev, dtype=dt)
        # sgs embedding + speaker anchor per row; codes are added each frame.
        self.base = torch.zeros(self.B, prep.H, device=dev, dtype=dt)
        self.fb_ids = torch.zeros(self.B, prep.n_vq, device=dev, dtype=torch.long)

    # Per-slot state; ``rows(n)`` slices these, everything else is shared.
    _ROW_STATE = ("h", "base", "fb_ids", "finished", "length", "caps")

    def rows(self, n: int):
        """This frame over slots ``[:n]`` only: views of the same state, the
        same counters (``frame``, the cache write index, the history clock)."""
        v = object.__new__(type(self))
        v.__dict__.update(self.__dict__)
        v.B = int(n)
        v.bb, v.samp, v.ac = self.bb.rows(n), self.samp.rows(n), self.ac.rows(n)
        v.rep = self.rep.rows(n) if self.rep is not None else None
        for name in self._ROW_STATE:
            if name in self.__dict__:
                setattr(v, name, getattr(self, name)[:n])
        if "codes" in self.__dict__:
            v.codes = self.codes[:, :n]
        return v

    def _feedback(self, codes: torch.Tensor) -> None:
        """Next backbone input = embeddings of [sgs, codes] + speaker anchor."""
        torch.add(codes, self.p.code_offsets, out=self.fb_ids)
        x = F.embedding_bag(self.fb_ids, self.p.emb_table, mode="sum") + self.base
        self.h.copy_(self.bb.step(x))

    def _acoustic(self):
        if self.rep is not None:
            self.rep.tables(self.samp)
        codes, is_eos = self.ac(self.h, self.samp, self.rep)
        if self.rep is not None:
            self.rep.add_frame(codes)
        return codes, is_eos

    @torch.no_grad()
    def _load_prompt(self, pf: Prefiller, b: int, cur: int, ids: torch.Tensor, spk) -> int:
        """Prefill ``ids`` and put it in slot ``b`` (write index ``cur``)."""
        with pf.lock:
            T, Tb = pf.run(ids, spk)
            self.bb.load_row(b, cur, T, pf.k[:, :, Tb - T:Tb], pf.v[:, :, Tb - T:Tb])
            self.h[b].copy_(pf.h_last)
            torch.add(self.p.sgs_emb, pf.spk_h, out=self.base[b])
        return T


class FastBatchFrame(_FrameBase):
    """Static batch (``engine.py``): rows are loaded together and the graph
    runs until every row has seen EOS or hit its cap. Sampling settings are
    fixed per graph (they key the graph cache), so top-k uses ``topk``."""

    def __init__(self, prep: Prepared, B: int, max_len: int, max_frames: int, *,
                 temperature: float, top_k: int, top_p: float, repetition_penalty: float,
                 repetition_window: int = DEFAULT_REP_WINDOW, warmup: int = 3):
        use_rep = not math.isclose(repetition_penalty, 1.0)
        super().__init__(prep, B, max_len, top_k_static=int(top_k), use_rep=use_rep,
                         repetition_window=repetition_window)
        dev = prep.device
        self.max_frames = int(max_frames)
        self.max_len = int(max_len)
        self.samp.set_all(temperature, top_k, top_p, repetition_penalty)
        self.codes = torch.zeros(self.max_frames, self.B, prep.n_vq, device=dev, dtype=torch.long)
        self.finished = torch.zeros(self.B, device=dev, dtype=torch.bool)
        self.length = torch.zeros(self.B, device=dev, dtype=torch.long)
        self.caps = torch.zeros(self.B, device=dev, dtype=torch.long)
        self.frame = torch.zeros((), device=dev, dtype=torch.long)
        self.all_done = torch.zeros((), device=dev, dtype=torch.bool)
        self._done_host = torch.zeros((), dtype=torch.bool, pin_memory=True)
        self.graph = _capture(self._body, dev, 1 if prep.warm else warmup)
        prep.warm = True

    def _body(self) -> None:
        codes, is_eos = self._acoustic()
        self.codes.index_copy_(0, torch.remainder(self.frame, self.max_frames).view(1), codes.unsqueeze(0))
        n = self.frame + 1
        done_now = ~self.finished & (is_eos | (n >= self.caps))
        self.length.copy_(torch.where(done_now, n, self.length))
        self.finished.logical_or_(done_now)
        self.all_done.copy_(self.finished.all())
        self.frame += 1
        self._feedback(codes)

    @torch.no_grad()
    def run(self, prompts: Sequence[torch.Tensor], spks: Sequence[Optional[torch.Tensor]],
            caps: Sequence[int], max_new_frames: int,
            on_frame: Optional[Callable[[], None]] = None) -> List[torch.Tensor]:
        """Generate for ``len(prompts) <= B`` rows (prompt ids from
        ``Prepared.prompt_ids``). Spare slots start finished. Returns per-row
        codes ``(T_b, n_vq)`` on the device."""
        n = len(prompts)
        if n > self.B:
            raise ValueError(f"{n} prompts for {self.B} slots")
        pf = Prefiller.of(self.p)
        T_max = max(int(t.shape[0]) for t in prompts)
        if T_max + 1 > self.max_len:
            raise ValueError(f"prompt of {T_max} tokens does not fit a cache of {self.max_len}")
        self.bb.reset()
        self.bb.cur.fill_(T_max)
        for b, (ids, spk) in enumerate(zip(prompts, spks)):
            self._load_prompt(pf, b, T_max, ids, spk)
        self.frame.zero_()
        self.length.zero_()
        self.all_done.zero_()
        self.finished.zero_()
        if n < self.B:
            self.finished[n:] = True
        cap_list = [min(int(c), self.max_frames) for c in caps] + [1] * (self.B - n)
        self.caps.copy_(torch.tensor(cap_list, dtype=torch.long), non_blocking=True)
        if self.rep is not None:
            self.rep.reset()
        steps = min(int(max_new_frames), self.max_frames)
        ev = None
        for f in range(steps):
            if on_frame is not None:
                on_frame()
            self.graph.replay()
            # Read "all done" one frame late: frame f+1 is queued before the
            # host waits for frame f's flag, so the GPU never idles on the
            # round trip. A finished batch runs at most one spare frame.
            if ev is not None:
                ev.synchronize()
                if bool(self._done_host):
                    break
            self._done_host.copy_(self.all_done, non_blocking=True)
            ev = torch.cuda.Event()
            ev.record()
        torch.cuda.current_stream().synchronize()
        frames = int(self.frame)
        length = torch.where(self.finished, self.length, self.frame.expand(self.B)).tolist()
        return [self.codes[:min(length[b], frames), b].clone() for b in range(n)]


class FastStreamFrame(_FrameBase):
    """Continuous batch (``stream.py``): ``B`` slots that rows enter
    (``admit_prompt``) and leave; per-row sampling, codes in a ring.
    Same public surface as ``stream.StreamFrame``.

    One graph per slot bucket (1, 2, 4, ..., B), each over views of slots
    ``[:n]``: ``replay(n_active)`` runs the smallest bucket that covers every
    busy slot, so a single listener does not pay for ``max_streams`` rows
    (RTX 5060 Ti: 3.0 ms instead of 3.9 ms a frame at 16 slots)."""

    def __init__(self, prep: Prepared, B: int, max_len: int, *,
                 repetition_window: int = DEFAULT_REP_WINDOW, codes_ring: int = 512, warmup: int = 3):
        super().__init__(prep, B, max_len, top_k_static=None, use_rep=True,
                         repetition_window=repetition_window)
        dev = prep.device
        self.max_len, self.ring = int(max_len), int(codes_ring)
        self.n_vq = prep.n_vq
        self.vocab = prep.V
        self.codes = torch.zeros(self.ring, self.B, prep.n_vq, device=dev, dtype=torch.long)
        self.finished = torch.ones(self.B, device=dev, dtype=torch.bool)
        self.length = torch.zeros(self.B, device=dev, dtype=torch.long)
        self.caps = torch.zeros(self.B, device=dev, dtype=torch.long)
        self.frame = torch.zeros((), device=dev, dtype=torch.long)
        self.host_frame = 0       # host mirror of ``frame`` (= replays since reset)
        self.buckets = [1 << i for i in range(self.B.bit_length()) if (1 << i) < self.B] + [self.B]
        pool = torch.cuda.graph_pool_handle()
        # The first capture warms kernels and handles; the smaller buckets
        # only need one eager run each (an eager frame is ~0.1 s on Windows).
        self.graphs = {n: _capture(self.rows(n)._body, dev, warmup if (i == 0 and not prep.warm) else 1, pool)
                       for i, n in enumerate(self.buckets)}
        prep.warm = True
        self.graph = self.graphs[self.B]
        self.reset()

    @torch.no_grad()
    def reset(self) -> None:
        self.h.zero_()
        self.codes.zero_()
        self.finished.fill_(True)
        self.length.zero_()
        self.caps.zero_()
        self.frame.zero_()
        self.host_frame = 0
        self.rep.reset()
        self.bb.reset()

    def _body(self) -> None:
        codes, is_eos = self._acoustic()
        self.codes.index_copy_(0, torch.remainder(self.frame, self.ring).view(1), codes.unsqueeze(0))
        n = self.frame + 1
        done_now = ~self.finished & (is_eos | (n >= self.caps))
        self.length.copy_(torch.where(done_now, n, self.length))
        self.finished.logical_or_(done_now)
        self.frame += 1
        self._feedback(codes)

    @torch.no_grad()
    def admit_prompt(self, b: int, ids: torch.Tensor, spk, cap: int, *, temperature: float, top_k: int,
                     top_p: float, repetition_penalty: float) -> int:
        """Fill slot ``b`` with a prompt; returns the global frame index of the
        row's first frame. ``cap`` is its own frame budget."""
        T = int(ids.shape[0])
        if T + cap + 2 > self.max_len:
            raise ValueError(f"prompt ({T} tokens) + {cap} frames exceeds the {self.max_len}-token cache")
        cur = self.host_frame          # write index = frames stepped since reset
        self._load_prompt(Prefiller.of(self.p), b, cur, ids, spk)
        self.rep.reset_rows(torch.tensor([b], device=self.h.device))
        self.samp.set_row(b, temperature, min(max(int(top_k), 0), self.vocab), top_p, repetition_penalty)
        f0 = self.host_frame
        self.caps[b] = f0 + int(cap)
        self.length[b] = 0
        self.finished[b] = False
        return f0

    def bucket(self, n_active: Optional[int]) -> int:
        """Smallest slot bucket covering slots ``[:n_active]`` (all for ``None``)."""
        if n_active is None:
            return self.B
        n = max(1, int(n_active))
        return next(b for b in self.buckets if b >= n)

    def replay(self, n_active: Optional[int] = None) -> None:
        """One frame for slots ``[:bucket(n_active)]``; no slot from
        ``n_active`` on may hold a running row."""
        self.graphs[self.bucket(n_active)].replay()
        self.host_frame += 1

    def status(self) -> List[int]:
        """``[finished(B) as 0/1..., length(B)...]`` in one host read."""
        return torch.cat([self.finished.long(), self.length]).tolist()

    def recent_codes(self, k: int, host: bool = True) -> torch.Tensor:
        f = self.host_frame
        idx = torch.remainder(torch.arange(f - k, f), self.ring).to(self.codes.device, non_blocking=True)
        blk = self.codes.index_select(0, idx)
        return blk.cpu() if host else blk
