"""
Benchmark of the GPU streaming path, one component at a time (issues #1-#2).

Fills the per-step rows of the before/after table:

* ``frame``  — one AR frame (graph replay, all slots) per ``--batch`` size
  (CUDA only);
* ``prefill`` — admitting ``--prompts`` prompts: HF prefill (what the stream
  scheduler used before) vs ``StaticBackbone.prefill`` (now); runs on CPU too;
* ``streams`` — in-process ``infer_stream`` from N threads starting together:
  TTFA median / max and worst RTF per ``--streams`` (CUDA only; the scheduler
  reads ``VIENEU_CODEC_GRAPH`` / ``VIENEU_CODEC_DTYPE`` / ``VIENEU_CODEC_OVERLAP``
  / ``VIENEU_STREAM_MAX_LEN``, so run it once per setting to compare).

    uv run python examples/bench_stream.py --json stream.json
    VIENEU_CODEC_GRAPH=0 uv run python examples/bench_stream.py --only streams --json before.json

The codec call alone is in ``examples/bench_codec.py``.
"""
import argparse
import json
import os
import platform
import statistics
import threading
import time
from pathlib import Path

import numpy as np
import torch

TEXT = ("Hôm nay trời đẹp quá, mình cùng nhau đi dạo một vòng quanh hồ "
        "rồi ghé quán cà phê quen nhé.")   # ~90 characters, like the docs' runs


def sync(dev):
    if dev.type == "cuda":
        torch.cuda.synchronize(dev)


def median_ms(fn, dev, warmup=3, iters=20):
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


def bench_frame(tts, sizes, iters):
    from vieneu.v3_turbo_serve.stream import StreamFrame
    model = tts.engine.model
    dev = next(model.parameters()).device
    out = []
    for B in sizes:
        frame = StreamFrame(model, B, int(model.config.max_position_embeddings))
        out.append({"B": B, "frame_ms": median_ms(frame.replay, dev, iters=iters)})
        print(f"  frame  B={B:<3} {out[-1]['frame_ms']:7.2f} ms")
        del frame
        if dev.type == "cuda":
            torch.cuda.empty_cache()
    return out


def bench_prefill(tts, counts, iters):
    from vieneu.v3_turbo_serve.fused import StaticBackbone
    eng = tts._get_batch_engine()
    dev = next(eng.model.parameters()).device
    voice = tts.get_preset_voice()
    req = {"text": TEXT, "speaker_emb": voice["speaker_emb"], "ref_codes": voice["codes"]}
    embed = eng._prompt_embeds(req)
    sb = StaticBackbone(eng.model, 1, 1024)   # prefill does not use the cache
    out = []
    with torch.no_grad():
        for n in counts:
            embeds = [embed] * n
            hf = median_ms(lambda: eng.bb.prefill(embeds), dev, iters=iters)
            own = median_ms(lambda: sb.prefill(embeds), dev, iters=iters)
            out.append({"prompts": n, "prompt_tokens": int(embed.shape[0]), "hf_ms": hf, "static_ms": own})
            print(f"  prefill n={n:<3} HF {hf:7.1f} ms | static {own:7.1f} ms")
    return out


def bench_streams(tts, counts):
    out = []
    for n in counts:
        res = [None] * n
        gate = threading.Barrier(n)

        def one(i):
            gate.wait()
            t0 = time.perf_counter()
            first, samples = None, 0
            for chunk in tts.infer_stream(TEXT, apply_watermark=False):
                if len(chunk) and first is None:
                    first = time.perf_counter() - t0
                samples += len(chunk)
            res[i] = (first, time.perf_counter() - t0, samples / 48000)

        ths = [threading.Thread(target=one, args=(i,)) for i in range(n)]
        for t in ths:
            t.start()
        for t in ths:
            t.join()
        ttfa = [r[0] * 1000 for r in res]
        rtf = [r[1] / r[2] for r in res if r[2]]
        out.append({"streams": n, "ttfa_median_ms": statistics.median(ttfa), "ttfa_max_ms": max(ttfa),
                    "rtf_max": max(rtf)})
        print(f"  streams N={n:<3} TTFA median {out[-1]['ttfa_median_ms']:6.0f} ms  max "
              f"{out[-1]['ttfa_max_ms']:6.0f} ms | worst RTF {out[-1]['rtf_max']:.2f}")
        time.sleep(0.5)
    return out


def env_info(dev):
    keys = ["VIENEU_CODEC_GRAPH", "VIENEU_CODEC_DTYPE", "VIENEU_CODEC_OVERLAP", "VIENEU_STREAM_MAX_LEN",
            "VIENEU_MAX_STREAMS"]
    info = {"torch": torch.__version__, "cuda": torch.version.cuda, "os": platform.platform(),
            "device": str(dev), "env": {k: os.environ[k] for k in keys if k in os.environ}}
    if dev.type == "cuda":
        p = torch.cuda.get_device_properties(dev)
        info.update(gpu=p.name, capability=f"{p.major}.{p.minor}", vram_gb=round(p.total_memory / 2**30, 1))
    return info


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--device", default="auto", help="auto | cuda | cpu")
    ap.add_argument("--only", nargs="+", default=["frame", "prefill", "streams"],
                    choices=["frame", "prefill", "streams"])
    ap.add_argument("--batch", type=int, nargs="+", default=[1, 8, 16, 32, 64], help="AR frame slots")
    ap.add_argument("--prompts", type=int, nargs="+", default=[1, 8, 16, 32], help="prompts per prefill")
    ap.add_argument("--streams", type=int, nargs="+", default=[1, 8, 16], help="concurrent infer_stream")
    ap.add_argument("--max-streams", type=int, default=16, help="scheduler slots for --only streams")
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--json", default=None)
    a = ap.parse_args()

    from vieneu import Vieneu
    device = "cpu" if a.device == "cpu" else ("cuda" if a.device == "cuda" or torch.cuda.is_available() else "cpu")
    tts = Vieneu(mode="v3turbo", device=device, backend="pytorch",
                 max_streams=max([a.max_streams] + a.streams))
    dev = next(tts.engine.model.parameters()).device
    info = env_info(dev)
    print(" | ".join(f"{k}={v}" for k, v in info.items()))
    results = {"env": info}
    if "frame" in a.only:
        results["frame"] = bench_frame(tts, a.batch, a.iters) if dev.type == "cuda" else "needs CUDA"
    if "prefill" in a.only:
        results["prefill"] = bench_prefill(tts, a.prompts, a.iters)
    if "streams" in a.only:
        if dev.type == "cuda":
            for _ in tts.infer_stream("Xin chào.", apply_watermark=False):   # warm-up: graph capture
                pass
            results["streams"] = bench_streams(tts, a.streams)
        else:
            results["streams"] = "needs CUDA"
    if a.json:
        Path(a.json).write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"→ {a.json}")
    tts.close()


if __name__ == "__main__":
    main()
