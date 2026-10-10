"""
Benchmark of the streaming codec call the GPU stream scheduler makes (issue #1).

Times one codec call — ``--frames`` code frames (4 = the scheduler's block)
for ``B`` slots, all active — through both paths, at every ``--max-streams``:

* ``old``: the codec's own ``batch_decode(streaming=True)`` session
  (what ``VIENEU_CODEC_GRAPH=0`` runs);
* ``new``: ``vieneu.v3_turbo_serve.codec_stream.StreamCodec`` (ring KV cache,
  CUDA graph on CUDA), once per ``--dtype``.

Then checks the audio on the preset voices' real codes: the new path against
the plain full decode (fp32 should match to ~1e-5), and every other dtype
against fp32 (correlation and SNR — listen too before trusting bf16).

    uv run python examples/bench_codec.py                          # CUDA if present
    uv run python examples/bench_codec.py --max-streams 8 16 32 64 --dtype fp32 bf16 --json codec.json

Only the codec is loaded (~88 MB), not the TTS model.
"""
import argparse
import json
import platform
import statistics
import time
from pathlib import Path

import torch

from vieneu.v3_turbo_serve.codec_stream import StreamCodec

ROOT = Path(__file__).resolve().parents[1]
MOSS_REPO = "OpenMOSS-Team/MOSS-Audio-Tokenizer-Nano"
DTYPES = {"fp32": None, "bf16": torch.bfloat16, "fp16": torch.float16}


def sync(dev):
    if dev.type == "cuda":
        torch.cuda.synchronize(dev)


def timed(fn, dev, warmup, iters):
    """Median wall time of ``fn()`` in ms (the GPU is synchronised around each call)."""
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(iters):
        sync(dev)
        t = time.perf_counter()
        fn()
        sync(dev)
        times.append((time.perf_counter() - t) * 1000)
    return statistics.median(times)


def bench_old(tok, B, frames, dev, warmup, iters):
    codes = torch.randint(0, 1024, (16, frames), device=dev)
    state = {"fresh": True}

    def call():
        out = tok.batch_decode([codes] * B, num_quantizers=16, streaming=True, max_batch_size=B,
                               reset_stream=state["fresh"])
        state["fresh"] = False
        out.audio.float().mean(1).cpu()   # what the scheduler does with it

    try:
        return timed(call, dev, warmup, iters)
    finally:
        tok._reset_batch_decode_streaming_state()


def bench_new(tok, B, frames, dev, dtype, warmup, iters):
    codec = StreamCodec(tok, B, compute_dtype=dtype)
    codes = torch.randint(0, 1024, (frames, B, 16), device=dev)
    active = torch.ones(frames, B, dtype=torch.bool, device=dev)
    ms = timed(lambda: codec.decode_block(codes, active), dev, warmup, iters)
    del codec
    if dev.type == "cuda":
        torch.cuda.empty_cache()
    return ms


def preset_codes(n):
    data = json.loads((ROOT / "src" / "vieneu" / "assets" / "voices_v3_turbo.json").read_text(encoding="utf-8"))
    out = []
    for name, v in data["presets"].items():
        if v.get("codes"):
            out.append((name, torch.tensor(v["codes"], dtype=torch.long)))
        if len(out) == n:
            break
    return out


def stream_decode(tok, codes, dtype):
    """One row through ``StreamCodec`` frame by frame → mono (T*3840,)."""
    dev = next(tok.parameters()).device
    codec = StreamCodec(tok, 1, compute_dtype=dtype)
    T = codes.shape[0]
    return codec.decode_block(codes.view(T, 1, -1).to(dev), torch.ones(T, 1, dtype=torch.bool, device=dev))[:, 0].reshape(-1)


def quality(tok, dtypes, n_voices):
    dev = next(tok.parameters()).device
    rows = []
    for name, codes in preset_codes(n_voices):
        with torch.no_grad():
            full = tok.decode(codes.T.unsqueeze(1).to(dev), return_dict=True).audio[0].float().mean(0).cpu()
        ref = stream_decode(tok, codes, None)
        row = {"voice": name, "frames": int(codes.shape[0]),
               "fp32_vs_full_max_abs": float((ref - full).abs().max())}
        for d in dtypes:
            if DTYPES[d] is None:
                continue
            got = stream_decode(tok, codes, DTYPES[d])
            err = got - ref
            row[f"{d}_corr"] = float(torch.corrcoef(torch.stack([ref, got]))[0, 1])
            row[f"{d}_snr_db"] = float(10 * torch.log10(ref.pow(2).sum() / err.pow(2).sum().clamp_min(1e-20)))
        rows.append(row)
    return rows


def env_info(dev):
    info = {"torch": torch.__version__, "cuda": torch.version.cuda, "python": platform.python_version(),
            "os": platform.platform(), "device": str(dev)}
    if dev.type == "cuda":
        p = torch.cuda.get_device_properties(dev)
        info.update(gpu=p.name, capability=f"{p.major}.{p.minor}", vram_gb=round(p.total_memory / 2**30, 1))
    return info


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--device", default="auto", help="auto | cuda | cpu")
    ap.add_argument("--max-streams", type=int, nargs="+", default=[8, 16, 32, 64], help="slots B to time")
    ap.add_argument("--frames", type=int, default=4, help="code frames per call (scheduler block = 4)")
    ap.add_argument("--dtype", nargs="+", default=["fp32", "bf16"], choices=list(DTYPES),
                    help="new-path dtypes (bf16/fp16 only apply on CUDA)")
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--no-old", action="store_true", help="skip the codec's own streaming decode")
    ap.add_argument("--voices", type=int, default=3, help="preset voices for the audio check (0 = skip)")
    ap.add_argument("--json", default=None, help="write the results here")
    a = ap.parse_args()

    dev = torch.device("cuda" if (a.device == "auto" and torch.cuda.is_available()) else
                       ("cpu" if a.device == "auto" else a.device))
    dtypes = [d for d in a.dtype if dev.type == "cuda" or DTYPES[d] is None]
    from transformers import AutoModel
    tok = AutoModel.from_pretrained(MOSS_REPO, trust_remote_code=True).to(dev).eval()
    info = env_info(dev)
    print(" | ".join(f"{k}={v}" for k, v in info.items()))

    results = []
    head = f"{'B':>4} | {'old ms':>8} | " + " | ".join(f"{'new ' + d + ' ms':>12}" for d in dtypes) + " | speedup"
    print(f"\ncodec call, {a.frames} frames, all slots active (median of {a.iters})\n" + head)
    with torch.no_grad():
        for B in a.max_streams:
            r = {"B": B, "frames": a.frames}
            r["old_ms"] = None if a.no_old else bench_old(tok, B, a.frames, dev, a.warmup, a.iters)
            for d in dtypes:
                r[f"new_{d}_ms"] = bench_new(tok, B, a.frames, dev, DTYPES[d], a.warmup, a.iters)
            best = min(r[f"new_{d}_ms"] for d in dtypes)
            sp = f"{r['old_ms'] / best:.1f}x" if r["old_ms"] else "-"
            old = f"{r['old_ms']:8.1f}" if r["old_ms"] else f"{'-':>8}"
            print(f"{B:>4} | {old} | " + " | ".join(f"{r[f'new_{d}_ms']:12.1f}" for d in dtypes) + f" | {sp}")
            results.append(r)

    q = quality(tok, dtypes, a.voices) if a.voices else []
    if q:
        print("\naudio check on preset voices (new path)")
        for row in q:
            extra = "  ".join(f"{d}: corr {row[f'{d}_corr']:.4f} SNR {row[f'{d}_snr_db']:.1f} dB"
                              for d in dtypes if f"{d}_corr" in row)
            print(f"  {row['voice']} ({row['frames']} frames): fp32 vs full decode max |diff| "
                  f"{row['fp32_vs_full_max_abs']:.1e}  {extra}")
    if a.json:
        Path(a.json).write_text(json.dumps({"env": info, "codec_call": results, "audio": q}, indent=2,
                                           ensure_ascii=False), encoding="utf-8")
        print(f"\n→ {a.json}")


if __name__ == "__main__":
    main()
