"""Tests cho đường CUDA-graph của v3 Turbo (``vieneu.v3_turbo_serve.fused``).

Chạy trong tiến trình riêng bởi ``tests/test_fused_frame.py`` (tên tệp cố ý không khớp ``test_*`` để pytest không tự gom vào tiến trình chính) — vài module
test cùng thư mục thay ``torch`` trong ``sys.modules`` bằng stub lúc import,
mà phần này cần torch thật. Lịch sử phạt lặp và sampler chạy được trên CPU
nên không cần GPU; phần backbone/graph chỉ chạy khi có CUDA (skip nếu không).
"""
import math
import random

import pytest

torch = pytest.importorskip("torch")

from vieneu._v3_turbo_engine.rep_history import RepetitionHistory
from vieneu.v3_turbo_serve.batched_acoustic import _sample_batched
from vieneu.v3_turbo_serve.fused import GpuRepHistory, batch_bucket, cache_len_bucket, sample_gpu


def _seen(h: GpuRepHistory, b: int, ch: int) -> set:
    return set(torch.nonzero(h.counts[b, ch] > 0).flatten().tolist())


@pytest.mark.parametrize("window", [0, 3, 8, 64])
def test_gpu_history_matches_deque_history(window):
    B, n_vq = 3, 4
    gpu = GpuRepHistory(B, n_vq, 1024, window, torch.device("cpu"))
    ref = [RepetitionHistory(n_vq, window) for _ in range(B)]
    rng = random.Random(7)
    for _ in range(50):
        for ch in range(n_vq):
            codes = torch.tensor([rng.randrange(12) for _ in range(B)])
            gpu.add(ch, codes)
            for b in range(B):
                ref[b][ch].add(int(codes[b]))
        gpu.advance()
        for ch in range(n_vq):
            for b in range(B):
                assert _seen(gpu, b, ch) == set(ref[b][ch])


def test_gpu_history_penalises_like_the_host_sampler():
    """Cùng logits, cùng lịch sử → cùng logits sau phạt (greedy để so được)."""
    B, n_vq, V = 2, 2, 32
    gpu = GpuRepHistory(B, n_vq, V, 4, torch.device("cpu"))
    ref = [RepetitionHistory(n_vq, 4) for _ in range(B)]
    for code in [(3, 5), (3, 9), (7, 5)]:
        c = torch.tensor(code)
        gpu.add(0, c)
        for b in range(B):
            ref[b][0].add(int(c[b]))
        gpu.advance()
    logits = torch.randn(B, V) * 3
    want = _sample_batched(logits.clone(), 0.0, 0, 1.0, 1.2, [ref[b][0] for b in range(B)])
    got = sample_gpu(gpu.penalise(logits.clone(), 0, 1.2), 0.0, 0, 1.0)
    assert torch.equal(want, got)


def test_reset_clears_everything():
    gpu = GpuRepHistory(1, 1, 16, 2, torch.device("cpu"))
    gpu.add(0, torch.tensor([5]))
    gpu.advance()
    gpu.reset()
    assert _seen(gpu, 0, 0) == set()
    assert int(gpu.frame) == 0


def test_sampler_respects_top_k_and_top_p():
    torch.manual_seed(0)
    logits = torch.tensor([[10.0, 9.0, -5.0, -6.0, -7.0]])
    for _ in range(50):
        assert int(sample_gpu(logits, 0.8, 2, 0.95)) in (0, 1)
    assert int(sample_gpu(logits, 0.0, 2, 0.95)) == 0


def test_buckets():
    assert [batch_bucket(n) for n in (1, 2, 3, 5, 8, 9, 16, 17)] == [1, 2, 4, 8, 8, 16, 16, 32]
    assert cache_len_bucket(700, 2048) == 1024
    assert cache_len_bucket(1025, 2048) == 1536
    assert cache_len_bucket(5000, 2048) == 2048


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_static_backbone_matches_hf_in_fp32():
    """Bước giải mã trên cache tĩnh phải khớp HF DynamicCache (fp32: ~1e-6)."""
    from vieneu import Vieneu
    from vieneu.v3_turbo_serve.fused import StaticBackbone

    tts = Vieneu(mode="v3turbo", device="cuda", backend="pytorch", dtype="float32")
    eng = tts._get_batch_engine()
    voice = tts.get_preset_voice()
    reqs = [{"text": t, "speaker_emb": voice["speaker_emb"], "ref_codes": voice["codes"]}
            for t in ["Xin chào các bạn.", "Hôm nay trời đẹp quá."]]
    embeds = [eng._prompt_embeds(r) for r in reqs]
    h, cache, mask, pos = eng.bb.prefill(embeds)
    sb = StaticBackbone(eng.model, len(embeds), 1024)
    sb.load(cache, mask, pos)
    H = eng.model.config.hidden_size
    for _ in range(4):
        x = torch.randn(len(embeds), 1, H, device=h.device, dtype=h.dtype) * 0.5
        h_ref, cache, mask, pos = eng.bb.decode_step(x, cache, mask, pos)
        h_new = sb.step(x)
        assert torch.allclose(h_ref, h_new, atol=1e-4, rtol=1e-4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_frame_hook_stops_a_running_batch_within_frames():
    """Một server dừng giữa chừng bằng cách raise từ ``frame_hook`` — phải ăn
    sau vài khung, không phải sau cả batch."""
    from vieneu import Vieneu

    tts = Vieneu(mode="v3turbo", device="cuda", backend="pytorch")
    eng = tts._get_batch_engine()
    calls = []

    def hook():
        calls.append(1)
        if len(calls) == 3:
            raise RuntimeError("cancelled")

    eng.frame_hook = hook
    with pytest.raises(RuntimeError, match="cancelled"):
        tts.infer_batch(["Xin chào các bạn, hôm nay trời đẹp quá."])  # giọng mặc định
    assert len(calls) == 3


# ── continuous batching (stream.py) ──────────────────────────────────────────

def test_gpu_history_rows_start_late_do_not_evict_unwritten_slots():
    """Row vào giữa chừng (``reset_rows``) chỉ được đuổi mã do chính nó ghi."""
    B, n_vq, window = 2, 1, 3
    gpu = GpuRepHistory(B, n_vq, 16, window, torch.device("cpu"))
    ref = [RepetitionHistory(n_vq, window) for _ in range(B)]
    for code in (1, 2, 3, 4):        # row 0 chạy trước 4 khung
        gpu.add(0, torch.tensor([code, 9]))
        gpu.advance()
        ref[0][0].add(code)
    gpu.reset_rows(torch.tensor([1]))   # row 1 bắt đầu tại khung 4
    assert _seen(gpu, 1, 0) == set()
    for code in (5, 6, 7, 8):
        gpu.add(0, torch.tensor([code, code]))
        gpu.advance()
        ref[0][0].add(code)
        ref[1][0].add(code)
        assert _seen(gpu, 0, 0) == set(ref[0][0])
        assert _seen(gpu, 1, 0) == set(ref[1][0])


def test_sample_gpu_rows_matches_scalar_sampler_per_row():
    """Mỗi row một bộ tham số: greedy row, top-k row, và row tắt top-k."""
    from vieneu.v3_turbo_serve.fused import sample_gpu_rows

    torch.manual_seed(0)
    logits = torch.tensor([[10.0, 9.0, -5.0, -6.0, -7.0]]).repeat(3, 1)
    temperature = torch.tensor([[0.0], [0.8], [0.8]])
    top_k = torch.tensor([[2], [2], [0]])
    top_p = torch.tensor([[0.95], [0.95], [1.0]])
    for _ in range(50):
        got = sample_gpu_rows(logits, temperature, top_k, top_p)
        assert int(got[0]) == 0
        assert int(got[1]) in (0, 1)
        assert int(got[2]) in range(5)
    # Row tắt top-k/top-p vẫn đúng phân phối: mã 2..4 gần như không bao giờ được chọn
    # với logits cách nhau 15 đơn vị, nhưng cấu trúc không cấm (không -inf).
    nan_logits = torch.full((1, 5), float("nan"))
    assert int(sample_gpu_rows(nan_logits, torch.tensor([[0.8]]), torch.tensor([[0]]), torch.tensor([[1.0]]))) in range(5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_load_row_matches_load_even_across_the_ring_boundary():
    """Nạp từng row vào cache ring (``load_row``) phải cho cùng bước giải mã
    như nạp cả batch (``load``), kể cả khi chỉ số ghi sắp quay vòng."""
    from vieneu import Vieneu
    from vieneu.v3_turbo_serve.fused import StaticBackbone

    tts = Vieneu(mode="v3turbo", device="cuda", backend="pytorch", dtype="float32")
    eng = tts._get_batch_engine()
    voice = tts.get_preset_voice()
    reqs = [{"text": t, "speaker_emb": voice["speaker_emb"], "ref_codes": voice["codes"]}
            for t in ["Xin chào các bạn.", "Hôm nay trời đẹp quá."]]
    embeds = [eng._prompt_embeds(r) for r in reqs]
    h, cache, mask, pos = eng.bb.prefill(embeds)
    T = mask.shape[1]
    H = eng.model.config.hidden_size
    for start in (0, 1024 - 7, 500):
        ref = StaticBackbone(eng.model, 2, 1024)
        ref.load(cache, mask, pos)
        sb = StaticBackbone(eng.model, 2, 1024)
        sb.cur.fill_(start)
        for i in range(2):
            Ti = embeds[i].shape[0]
            keys = [(cache.layers[l].keys if hasattr(cache, "layers") else cache.key_cache[l])[i, :, T - Ti:T]
                    for l in range(len(sb.layers))]
            vals = [(cache.layers[l].values if hasattr(cache, "layers") else cache.value_cache[l])[i, :, T - Ti:T]
                    for l in range(len(sb.layers))]
            sb.load_row(i, keys, vals)
        for _ in range(12):
            x = torch.randn(2, 1, H, device="cuda") * 0.5
            assert torch.allclose(ref.step(x), sb.step(x), atol=1e-4, rtol=1e-4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_infer_stream_serves_concurrent_callers_and_matches_full_decode():
    """Nhiều thread gọi ``infer_stream`` cùng lúc: mỗi luồng nhận audio đúng
    độ dài mã của mình, audio stream khớp decode đầy đủ cùng mã, huỷ giữa
    chừng giải phóng slot."""
    import threading
    import numpy as np
    from vieneu import Vieneu

    tts = Vieneu(mode="v3turbo", device="cuda", backend="pytorch", max_streams=4)
    sched = tts._get_stream_scheduler()
    assert sched is not None
    texts = ["Xin chào các bạn.", "Hôm nay trời đẹp quá, mình đi dạo nhé.", "Một hai ba bốn năm."]
    out = [None] * len(texts)

    def run(i):
        out[i] = np.concatenate(list(tts.infer_stream(texts[i], apply_watermark=False)))

    ths = [threading.Thread(target=run, args=(i,)) for i in range(len(texts))]
    for t in ths:
        t.start()
    for t in ths:
        t.join()
    for a in out:
        assert a.ndim == 1 and len(a) % 3840 == 0 and 0.5 <= len(a) / 48000 <= 6.0
        assert np.abs(a).max() > 0.05
    assert sched.n_active == 0

    # Audio stream == decode đầy đủ của cùng mã (codec streaming không lookahead).
    spk, codes = tts._resolve_ref(None, None, True, True)
    from vieneu_utils.phonemize_text import phonemize_text_with_emotions
    h = sched.submit(phonemes=phonemize_text_with_emotions(texts[1]), speaker_emb=spk,
                     ref_codes=codes, max_new_frames=80)
    streamed = np.concatenate(list(h))
    idx = torch.remainder(torch.arange(h.f0, h.f0 + h.frames, device="cuda"), sched.frame.ring)
    row_codes = sched.frame.codes.index_select(0, idx)[:, h.slot].cpu()
    full = _full_decode(tts.engine.audio_tokenizer, row_codes).numpy()
    assert len(streamed) == len(full) == h.frames * 3840
    assert np.abs(streamed - full).max() < 1e-3

    g = tts.infer_stream(texts[1], apply_watermark=False)
    next(g)
    g.close()
    import time
    time.sleep(0.5)
    assert sched.n_active == 0
    tts.close()


# ── issue #198: reference encoder must not append the audible pad frame ──────

@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_pytorch_reference_encoder_has_no_pad_frame():
    """Clip lẻ mẫu → đúng ceil(n/3840) frame, frame cuối là mã thật (im lặng = 482),
    không phải mã đệm 455 nghe được."""
    import numpy as np
    from vieneu import Vieneu
    from vieneu_utils.core_utils import CODEC_SAMPLES_PER_FRAME as F, ENCODER_PAD_CODE

    tts = Vieneu(mode="v3turbo", device="cuda", backend="pytorch")
    enc = lambda w: tts.engine._encode_ref_wav(torch.from_numpy(w).unsqueeze(0), 48_000)
    for n, extra in [(10, 0), (50, 1), (50, F - 1)]:
        codes = enc(np.zeros(n * F + extra, np.float32))
        assert codes.shape[0] == n + (1 if extra else 0)
        assert int(codes[-1, 0]) != ENCODER_PAD_CODE and int(codes[-1, 0]) == 482
    for v in tts._preset_voices.values():
        if v["codes"] is not None:
            assert int(v["codes"][-1, 0]) != ENCODER_PAD_CODE


def test_onnx_reference_encoder_has_no_pad_frame():
    """Đường ONNX từng LUÔN trả n+1 frame với frame cuối 455, kể cả clip tròn frame."""
    import numpy as np
    from vieneu_utils.core_utils import CODEC_SAMPLES_PER_FRAME as F, ENCODER_PAD_CODE
    try:
        from vieneu import Vieneu
        tts = Vieneu(mode="v3turbo", device="cpu", backend="onnx")
    except Exception as e:   # noqa: BLE001 — no model cache / no onnxruntime here
        pytest.skip(f"ONNX engine unavailable: {e}")
    for n, extra in [(10, 0), (50, 1), (50, F - 1)]:
        codes = tts.engine._encode_ref_wav(np.zeros(n * F + extra, np.float32), 48_000)
        assert codes.shape[0] == n + (1 if extra else 0)
        assert int(codes[-1, 0]) != ENCODER_PAD_CODE and int(codes[-1, 0]) == 482


# ── streaming codec: ring KV cache + CUDA graph (codec_stream.py) ────────────

_MOSS_REPO = "OpenMOSS-Team/MOSS-Audio-Tokenizer-Nano"


def _moss_codec(device: str):
    """Codec thật; trên CPU chỉ dùng bản đã có trong cache HF (không tải 88 MB
    trong CI) trừ khi đặt VIENEU_TEST_CODEC=1."""
    import os
    from transformers import AutoModel
    local_only = device == "cpu" and os.environ.get("VIENEU_TEST_CODEC") != "1"
    try:
        tok = AutoModel.from_pretrained(_MOSS_REPO, trust_remote_code=True, local_files_only=local_only)
    except Exception as e:   # noqa: BLE001 — chưa có trong cache / không có mạng
        pytest.skip(f"MOSS codec unavailable: {e}")
    return tok.to(device).eval()


def _full_decode(tok, codes_tn: torch.Tensor) -> torch.Tensor:
    """Decode thường (không streaming) của mã ``(T, n_vq)`` → mono ``(T*3840,)``.

    Tắt TF32 của cuDNN khi decode: các conv 1x1 của quantizer chạy TF32 (mặc
    định của PyTorch) lệch tới ~3e-3 so với fp32 thật trên GPU Ampere+, tuỳ
    thuật toán cuDNN chọn theo batch — mốc so sánh phải là fp32 thật."""
    tf32 = torch.backends.cudnn.allow_tf32
    torch.backends.cudnn.allow_tf32 = False
    try:
        with torch.no_grad():
            out = tok.decode(codes_tn.T.unsqueeze(1).to(next(tok.parameters()).device), return_dict=True)
    finally:
        torch.backends.cudnn.allow_tf32 = tf32
    return out.audio[0].float().mean(0).cpu()


def test_plan_codec_block_windows():
    from vieneu.v3_turbo_serve.stream import plan_codec_block
    # Ranh giới K=4 (base=8, tick=12): hai row đầy đủ, một row mới vào ở frame 10.
    first, active = plan_codec_block([(0, 8, 12), (1, 8, 12), (2, 10, 12)], base=8)
    assert first == 0 and active == [{0, 1}, {0, 1}, {0, 1, 2}, {0, 1, 2}]
    # Lead-in giữa block: row cũ đã decode tới 10, row mới (slot 3) từ 9.
    first, active = plan_codec_block([(0, 10, 11), (3, 9, 11)], base=7)
    assert first == 2 and active == [{3}, {0, 3}]
    # Row kết thúc đúng ranh giới: không còn gì để decode.
    assert plan_codec_block([(0, 12, 12)], base=8) == (0, [])
    # Row kết thúc giữa block: chỉ decode tới frame cuối của nó.
    first, active = plan_codec_block([(5, 8, 10)], base=8)
    assert first == 0 and active == [{5}, {5}]
    with pytest.raises(ValueError):
        plan_codec_block([(0, 3, 6)], base=4)


def _run_slots(codec, codes, rows):
    """Đẩy ``codes`` (N, B, n_vq) qua codec từng frame; ``rows`` = (slot, start,
    len). Slot của row xong được reset trước khi row sau dùng lại."""
    N, B, _ = codes.shape
    active = torch.zeros(N, B, dtype=torch.bool)
    for s, a, n in rows:
        active[a:a + n, s] = True
    dev = codec.device
    out = []
    for f in range(N):
        codec.reset_slots([s for s, a, n in rows if a + n == f])
        out.append(codec.decode_block(codes[f:f + 1].to(dev), active[f:f + 1].to(dev))[0])
    return torch.stack(out)   # (N, B, 3840)


# 170 frame: cửa sổ mọi tầng đều quay vòng (tầng đầu: 500 vị trí / 4 mỗi frame = 125 frame).
_ROWS = [(0, 0, 170), (1, 5, 40), (2, 30, 120), (1, 60, 100)]


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_stream_codec_matches_full_decode(device):
    """Nhiều slot vào/ra ở các thời điểm khác nhau, slot rảnh, slot được dùng
    lại, cửa sổ attention quay vòng: audio từng row == decode đầy đủ mã của nó."""
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    from vieneu.v3_turbo_serve.codec_stream import StreamCodec
    tok = _moss_codec(device)
    torch.manual_seed(1)
    codes = torch.randint(0, 1024, (170, 3, 16))
    out = _run_slots(StreamCodec(tok, 3), codes, _ROWS)   # CUDA: qua CUDA graph
    for s, a, n in _ROWS:
        full = _full_decode(tok, codes[a:a + n, s])
        got = out[a:a + n, s].reshape(-1)
        assert got.shape == full.shape
        assert (got - full).abs().max().item() < 1e-3, (s, a, n)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_stream_codec_graph_matches_eager():
    from vieneu.v3_turbo_serve.codec_stream import StreamCodec
    tok = _moss_codec("cuda")
    torch.manual_seed(2)
    codes = torch.randint(0, 1024, (40, 4, 16))
    rows = [(0, 0, 40), (1, 3, 20), (3, 10, 30), (1, 25, 15)]
    graphed = StreamCodec(tok, 4, use_graph=True)
    eager = StreamCodec(tok, 4, use_graph=False)
    assert graphed.graph is not None and eager.graph is None
    a, b = _run_slots(graphed, codes, rows), _run_slots(eager, codes, rows)
    for s, st, n in rows:
        assert (a[st:st + n, s] - b[st:st + n, s]).abs().max().item() < 1e-4


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_stream_codec_bf16_stays_close_to_fp32():
    """bf16 (VIENEU_CODEC_DTYPE=bf16) trên mã giọng thật: chỉ chặn hỏng nặng —
    chất lượng nghe được phải kiểm bằng tai / examples/bench_codec.py."""
    import json
    from pathlib import Path
    from vieneu.v3_turbo_serve.codec_stream import StreamCodec
    tok = _moss_codec("cuda")
    path = Path(__file__).resolve().parents[2] / "src" / "vieneu" / "assets" / "voices_v3_turbo.json"
    v = next(iter(json.loads(path.read_text(encoding="utf-8"))["presets"].values()))
    codes = torch.tensor(v["codes"], dtype=torch.long)        # (T, 16), giọng thật
    rows = [(0, 0, codes.shape[0])]
    c = codes.unsqueeze(1)
    ref = _run_slots(StreamCodec(tok, 1), c, rows)[:, 0].reshape(-1)
    bf = _run_slots(StreamCodec(tok, 1, compute_dtype=torch.bfloat16), c, rows)[:, 0].reshape(-1)
    corr = torch.corrcoef(torch.stack([ref, bf]))[0, 1].item()
    assert corr > 0.95, corr


class _FakeStreamFrame:
    """Thay ``StreamFrame`` (cần CUDA) cho test sổ sách của scheduler: mỗi
    ``replay()`` ghi frame kế tiếp của từng row vào ring mã như graph thật."""

    def __init__(self, B: int, n_vq: int = 16, ring: int = 512):
        self.B, self.n_vq, self.ring = B, n_vq, ring
        self.codes = torch.zeros(ring, B, n_vq, dtype=torch.long)
        self.frame = 0
        self.rows = {}      # slot -> (f0, codes (n, n_vq))
        self.finished = [True] * B
        self.length = [0] * B

    def admit(self, b, codes):
        self.rows[b] = (self.frame, codes)
        self.finished[b] = False
        return self.frame

    def replay(self):
        f = self.frame
        for b, (f0, c) in self.rows.items():
            if not self.finished[b]:
                self.codes[f % self.ring, b] = c[f - f0]
                if f + 1 - f0 >= c.shape[0]:
                    self.finished[b], self.length[b] = True, f + 1
        self.frame += 1

    def status(self):
        return [int(x) for x in self.finished] + list(self.length)

    def recent_codes(self, k, host=True):
        idx = torch.remainder(torch.arange(self.frame - k, self.frame), self.ring)
        return self.codes.index_select(0, idx)


@pytest.mark.parametrize("use_stream_codec, overlap", [(True, False), (True, True), (False, False)])
def test_scheduler_decode_bookkeeping_matches_full_decode(use_stream_codec, overlap):
    """``_step``/``_decode`` thật của scheduler (lead-in, kết thúc giữa block,
    slot dùng lại) với graph giả: audio mỗi luồng == decode đầy đủ mã của nó,
    qua StreamCodec (cả chế độ overlap: audio giao trễ một tick) lẫn session
    streaming của chính codec."""
    from vieneu.v3_turbo_serve.codec_stream import StreamCodec
    from vieneu.v3_turbo_serve.stream import StreamHandle, V3TurboStreamScheduler
    tok = _moss_codec("cpu")
    B = 3
    sched = object.__new__(V3TurboStreamScheduler)   # no __init__: no CUDA graph, no worker thread
    sched.B, sched.K, sched.leadin, sched.tick = B, 4, 2, 0
    sched.tok = tok
    sched.frame = _FakeStreamFrame(B)
    sched._slots, sched._codec_rows, sched._codec_fresh = [None] * B, [], True
    sched.codec = StreamCodec(tok, B) if use_stream_codec else None
    sched.overlap, sched._inflight = overlap, None
    torch.manual_seed(3)
    # (tick at which it arrives, frames) — row 3 reuses a freed slot; lengths not multiples of K.
    plan = [(0, 13), (1, 22), (6, 7), (15, 18)]
    codes = [torch.randint(0, 1024, (n, 16)) for _, n in plan]
    handles = [None] * len(plan)
    for _ in range(200):
        for i, (t, _n) in enumerate(plan):
            if handles[i] is None and t <= sched.tick and None in sched._slots:   # else it waits, as in the queue
                b = sched._slots.index(None)
                h = StreamHandle(sched, {})
                h.slot, h.f0 = b, sched.frame.admit(b, codes[i])
                h.decoded_upto = h.f0
                sched._slots[b] = h
                sched._codec_rows.append(h)
                handles[i] = h
        if all(h is not None for h in handles) and not sched._codec_rows:
            break
        sched._step()
    sched._flush(wait=True)      # what the worker does before going idle
    assert not sched._codec_rows and sched._slots == [None] * B and sched._inflight is None
    for i, h in enumerate(handles):
        chunks = []
        while True:
            item = h.q.get_nowait()
            if item is None:
                break
            chunks.append(item)
        got = torch.from_numpy(__import__("numpy").concatenate(chunks))
        full = _full_decode(tok, codes[i])
        assert got.shape == full.shape, (i, got.shape, full.shape)
        assert (got - full).abs().max().item() < 1e-3, i


# ── issue #4: dtype="auto" must not pick emulated bf16 ───────────────────────

@pytest.mark.parametrize("native, want", [(True, torch.bfloat16), (False, torch.float32)])
def test_auto_dtype_needs_native_bf16(monkeypatch, native, want):
    """Turing/Volta: torch.cuda.is_bf16_supported() mặc định trả True nhờ giả lập;
    dtype="auto" chỉ được chọn bf16 khi card có bf16 thật (Ampere+)."""
    from types import SimpleNamespace
    from vieneu._v3_turbo_engine.inference_v3_turbo import VieNeuTTSv3Turbo
    monkeypatch.setattr(torch.cuda, "is_bf16_supported",
                        lambda including_emulation=True: native or including_emulation)
    fake = SimpleNamespace(device=torch.device("cuda"))
    assert VieNeuTTSv3Turbo._resolve_dtype(fake, "auto") == want
    assert VieNeuTTSv3Turbo._resolve_dtype(SimpleNamespace(device=torch.device("cpu")), "auto") == torch.float32


# ── issue #3: batch path — batched codec decode, coalesced concurrent calls ──

def test_decode_rows_matches_one_decode_per_row():
    """Nhiều row độ dài khác nhau (cả row rỗng), ngân sách nhỏ để buộc chia
    nhóm: kết quả == decode từng row riêng như trước."""
    from vieneu.v3_turbo_serve.engine import decode_rows
    tok = _moss_codec("cpu")
    torch.manual_seed(4)
    lens = [37, 0, 12, 60, 59, 5]
    codes = [torch.randint(0, 1024, (n, 16)) for n in lens]
    got = decode_rows(tok, codes, budget_mb=60)   # 60-frame row ≈ 14 MB → nhóm ≤ 4 row
    for c, g in zip(codes, got):
        if len(c) == 0:
            assert g.shape == (0,)
            continue
        want = _full_decode(tok, c).numpy()
        assert g.shape == want.shape
        assert abs(g - want).max() < 1e-4


def _coalescer_threads(co, calls, fn):
    """Start one thread per call, each only once the previous one is queued,
    so the queue order is the order of ``calls``."""
    import threading
    import time
    out = [None] * len(calls)

    def one(i, key, items):
        try:
            out[i] = co.run(key, items, fn)
        except Exception as e:   # noqa: BLE001
            out[i] = e
    ths = []
    for i, (k, it) in enumerate(calls):
        n = len(co._queue)
        t = threading.Thread(target=one, args=(i, k, it))
        t.start()
        ths.append(t)
        while len(co._queue) == n:
            time.sleep(0.005)
    return ths, out


def test_coalescer_merges_waiting_calls_with_the_same_key():
    import threading
    import time
    from vieneu.v3_turbo_serve.engine import _Coalescer
    co = _Coalescer(max_items=5)
    runs = []
    release = threading.Event()

    def fn(items):
        runs.append(list(items))
        if len(runs) == 1:
            release.wait(5)          # run đầu giữ máy bận để các call sau xếp hàng
        return [x * 10 for x in items]

    first = threading.Thread(target=lambda: co.run("a", [1], fn))
    first.start()
    while not runs:
        time.sleep(0.01)
    calls = [("a", [2, 3]), ("b", [4]), ("a", [5]), ("a", [6, 7, 8])]
    ths, out = _coalescer_threads(co, calls, fn)   # tất cả đã vào hàng đợi, đúng thứ tự
    release.set()
    first.join(5)
    for t in ths:
        t.join(5)
    assert out == [[20, 30], [40], [50], [60, 70, 80]]
    # Sau run đầu: [2,3]+[5] chung một run ("b" khác key, [6,7,8] vượt trần 5),
    # rồi "b", rồi [6,7,8].
    assert runs[1:] == [[2, 3, 5], [4], [6, 7, 8]] and co.runs == 4


def test_coalescer_isolates_a_failing_call():
    import threading
    import time
    from vieneu.v3_turbo_serve.engine import _Coalescer
    co = _Coalescer(max_items=8)
    release = threading.Event()
    seen = []

    def fn(items):
        seen.append(list(items))
        if len(seen) == 1:
            release.wait(5)
        if "bad" in items:
            raise ValueError("bad input")
        return [x.upper() for x in items]

    first = threading.Thread(target=lambda: co.run("k", ["x"], fn))
    first.start()
    while not seen:
        time.sleep(0.01)
    ths, out = _coalescer_threads(co, [("k", ["a", "b"]), ("k", ["bad"]), ("k", ["c"])], fn)
    release.set()
    first.join(5)
    for t in ths:
        t.join(5)
    assert out[0] == ["A", "B"] and out[2] == ["C"]
    assert isinstance(out[1], ValueError)


# ── issue #2: AR frame — static acoustic step, own prefill into the ring ─────

def _tts_fp32(device: str):
    """Model v3 Turbo fp32; trên CPU chỉ chạy khi model đã có trong cache HF."""
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    if device == "cpu":
        from huggingface_hub import try_to_load_from_cache
        if not isinstance(try_to_load_from_cache("pnnbao-ump/VieNeu-TTS-v3-Turbo", "update/model.safetensors"), str):
            pytest.skip("v3 Turbo weights not in the local HF cache")
    from vieneu import Vieneu
    return Vieneu(mode="v3turbo", device=device, backend="pytorch", dtype="float32")


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_acoustic_static_steps_match_cached_step(device):
    """``acoustic_frame_gpu`` (buffer K/V cố định, bias dựng sẵn) == đường
    ``cached_step`` cũ: greedy cho đúng cùng mã và cùng cờ EOS."""
    from vieneu.v3_turbo_serve.batched_acoustic import generate_frame_batched
    from vieneu.v3_turbo_serve.fused import acoustic_frame_gpu
    m = _tts_fp32(device).engine.model
    torch.manual_seed(0)
    for B in (1, 3):
        h = torch.randn(B, m.config.hidden_size, device=device) * 0.5
        want, pre = generate_frame_batched(m, h, temperature=0.0, top_k=25, top_p=0.95)
        want_eos = m.text_lm_head(pre[:, 0]).float().argmax(-1) == m.config.speech_generation_end_token_id
        got, eos = acoustic_frame_gpu(m, h, temperature=0.0, top_k=25, top_p=0.95,
                                      repetition_penalty=1.0, rep=None)
        assert torch.equal(want, got) and torch.equal(want_eos, eos)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_static_prefill_matches_hf_prefill(device):
    """``StaticBackbone.prefill`` (không DynamicCache) == prefill HF: hidden cuối
    và K/V từng row (độ dài khác nhau, có row không mã tham chiếu)."""
    from vieneu.v3_turbo_serve.fused import StaticBackbone
    tts = _tts_fp32(device)
    eng = tts._get_batch_engine()
    voice = tts.get_preset_voice()
    reqs = [{"text": t, "speaker_emb": voice["speaker_emb"], "ref_codes": voice["codes"]}
            for t in ["Xin chào các bạn.", "Hôm nay trời đẹp quá, mình đi dạo nhé.", "Ba."]]
    reqs.append({"text": "Không có mã tham chiếu.", "speaker_emb": voice["speaker_emb"],
                 "ref_codes": voice["codes"], "use_ref_codes": False})
    embeds = [eng._prompt_embeds(r) for r in reqs]
    h_ref, cache, mask, _ = eng.bb.prefill(embeds)
    sb = StaticBackbone(eng.model, len(embeds), 1024)
    h_new, keys, values = sb.prefill(embeds)
    assert torch.allclose(h_ref, h_new, atol=1e-4, rtol=1e-4)
    T = mask.shape[1]
    for r, e in enumerate(embeds):
        Ti = e.shape[0]
        for l in range(len(sb.layers)):
            layer = cache.layers[l] if hasattr(cache, "layers") else None
            k = (layer.keys if layer is not None else cache.key_cache[l])[r, :, T - Ti:]
            v = (layer.values if layer is not None else cache.value_cache[l])[r, :, T - Ti:]
            assert keys[r][l].shape == k.shape
            assert torch.allclose(k, keys[r][l], atol=1e-4) and torch.allclose(v, values[r][l], atol=1e-4)


# ── fast path (fastpath.py, codec_stream fast stages): same maths, fewer kernels ──

def _fast_reqs(tts):
    voice = tts.get_preset_voice()
    reqs = [{"text": t, "speaker_emb": voice["speaker_emb"], "ref_codes": voice["codes"]}
            for t in ["Xin chào các bạn.", "Hôm nay trời đẹp quá, mình đi dạo nhé.", "Ba."]]
    reqs.append({"text": "Không có mã tham chiếu.", "speaker_emb": voice["speaker_emb"],
                 "ref_codes": voice["codes"], "use_ref_codes": False})
    return reqs


def test_fast_sampler_top_k_then_top_p_greedy_and_nan():
    """Top-k rồi top-p trên phân phối top-k đã chuẩn hoá lại (quy tắc của
    ``_sample_token``), greedy khi temperature 0, top-k riêng từng row, và mã
    luôn nằm trong [0, V) kể cả khi logits là NaN (slot chết của batch liên tục)."""
    from vieneu.v3_turbo_serve.fastpath import RowSampling, Sampler
    torch.manual_seed(0)
    V = 8
    raw = torch.tensor([[10.0, 9.0, 8.5, -5.0, -6.0, -7.0, -8.0, -9.0]])
    hyp = torch.ones(1, 1)
    samp = RowSampling(1, V, "cpu")
    samp.set_all(0.8, 2, 0.95, 1.0)
    s2 = Sampler(1, V, 2, "cpu")
    assert {int(s2(raw, hyp, 0, samp, None)) for _ in range(200)} == {0, 1}
    # top-3 renormalised = (0.63, 0.23, 0.14): top_p=0.3 keeps only the best.
    samp.set_all(1.0, 3, 0.3, 1.0)
    s3 = Sampler(1, V, 3, "cpu")
    assert {int(s3(raw, hyp, 0, samp, None)) for _ in range(50)} == {0}
    samp.set_all(0.0, 25, 0.95, 1.0)
    assert int(Sampler(1, V, 25, "cpu")(raw, hyp, 0, samp, None)) == 0
    rows = RowSampling(2, V, "cpu")
    rows.set_row(0, 0.8, 1, 1.0, 1.0)     # top-1
    rows.set_row(1, 0.8, 0, 1.0, 1.0)     # top-k off
    sr = Sampler(2, V, None, "cpu")
    got = torch.stack([sr(raw.repeat(2, 1), torch.ones(2, 1), 0, rows, None) for _ in range(300)])
    assert set(got[:, 0].tolist()) == {0}
    assert {0, 1, 2} <= set(got[:, 1].tolist())
    out = sr(torch.full((2, V), float("nan")), torch.ones(2, 1), 0, rows, None)
    assert ((out >= 0) & (out < V)).all()


def test_fast_rep_matches_gpu_history():
    """``FastRep`` (một lần mỗi frame cho mọi codebook) == ``GpuRepHistory``,
    kể cả row vào giữa chừng."""
    from vieneu.v3_turbo_serve.fastpath import FastRep
    B, n_vq, V, window = 2, 3, 16, 4
    fast = FastRep(B, n_vq, V, window, torch.device("cpu"))
    ref = GpuRepHistory(B, n_vq, V, window, torch.device("cpu"))
    rng = random.Random(5)
    for f in range(20):
        if f == 7:
            fast.reset_rows(torch.tensor([1]))
            ref.reset_rows(torch.tensor([1]))
        codes = torch.tensor([[rng.randrange(6) for _ in range(n_vq)] for _ in range(B)])
        fast.add_frame(codes)
        for ch in range(n_vq):
            ref.add(ch, codes[:, ch])
        ref.advance()
        assert torch.equal(fast.counts, ref.counts), f


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_baddbmm32_expands_a_broadcast_bias():
    """torch 2.8: ``baddbmm(out_dtype=...)`` không broadcast bias mà trả về
    batch của bias — ``_baddbmm32`` phải tự expand."""
    from vieneu.v3_turbo_serve.fastpath import _baddbmm32
    a = torch.randn(6, 2, 8, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(6, 8, 3, device="cuda", dtype=torch.bfloat16)
    bias = torch.randn(1, 2, 3, device="cuda")
    got = _baddbmm32(bias, a, b, 0.5)
    assert got.shape == (6, 2, 3) and got.dtype == torch.float32
    assert torch.allclose(got, bias + 0.5 * (a.float() @ b.float()), atol=1e-3, rtol=1e-3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_fast_prefill_and_step_match_the_fused_path_in_fp32():
    """Prefill bằng graph (``Prefiller``, embedding trong graph) và bước giải mã
    ``FastBackbone`` == ``StaticBackbone`` (fp32: ~1e-6)."""
    from vieneu.v3_turbo_serve import fastpath as FP
    from vieneu.v3_turbo_serve.fused import StaticBackbone
    tts = _tts_fp32("cuda")
    eng = tts._get_batch_engine()
    reqs = _fast_reqs(tts)
    n = len(reqs)
    embeds = [eng._prompt_embeds(r) for r in reqs]
    h_ref, keys, values = StaticBackbone(eng.model, n, 1024).prefill(embeds)
    prep = FP.Prepared.of(eng.model)
    pf = FP.Prefiller.of(prep)
    new, old = FP.FastBackbone(prep, n, 1024), StaticBackbone(eng.model, n, 1024)
    for r, req in enumerate(reqs):
        ids, spk = eng._prompt_ids(req)
        with pf.lock:
            T, Tb = pf.run(ids, spk)
            assert T == embeds[r].shape[0]
            assert torch.allclose(pf.h_last, h_ref[r], atol=1e-4, rtol=1e-4)
            for l in range(prep.n_layers):
                assert torch.allclose(pf.k[l, :, Tb - T:Tb], keys[r][l], atol=1e-4)
                assert torch.allclose(pf.v[l, :, Tb - T:Tb], values[r][l], atol=1e-4)
            new.load_row(r, 0, T, pf.k[:, :, Tb - T:Tb], pf.v[:, :, Tb - T:Tb])
        old.load_row(r, keys[r], values[r])
    H = prep.H
    for _ in range(6):
        x = torch.randn(n, H, device="cuda") * 0.5
        assert torch.allclose(new.step(x), old.step(x.view(n, 1, H)), atol=1e-4, rtol=1e-4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_fast_greedy_generation_matches_the_fused_path_in_fp32():
    """Cả vòng sinh greedy (phạt lặp bật) qua ``FastBatchFrame`` == ``FusedFrame``:
    cùng mã từng frame, cùng độ dài."""
    from vieneu.v3_turbo_serve import fastpath as FP
    from vieneu.v3_turbo_serve.fused import FusedFrame
    tts = _tts_fp32("cuda")
    eng = tts._get_batch_engine()
    reqs = _fast_reqs(tts)
    samp = dict(temperature=0.0, top_k=25, top_p=0.95, repetition_penalty=1.2, repetition_window=64)
    embeds = [eng._prompt_embeds(r) for r in reqs]
    h, cache, mask, pos = eng.bb.prefill(embeds)
    spk = torch.cat([tts.engine._resolve_speaker_emb(r["speaker_emb"]) for r in reqs])
    want = FusedFrame(eng.model, 4, 1024, 512, **samp).run(h, cache, mask, pos, [200] * 4, spk, 200)
    prompts = [eng._prompt_ids(r) for r in reqs]
    got = FP.FastBatchFrame(FP.Prepared.of(eng.model), 4, 1024, 512, **samp).run(
        [p[0] for p in prompts], [p[1] for p in prompts], [200] * 4, 200)
    for a, b in zip(want, got):
        assert a.shape == b.shape and torch.equal(a.cpu(), b.cpu())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_stream_codec_slot_buckets_match_full_width():
    """Chỉ tính các slot bận (bucket theo ``n_active``) cho đúng audio như tính đủ B slot."""
    from vieneu.v3_turbo_serve.codec_stream import StreamCodec
    tok = _moss_codec("cuda")
    torch.manual_seed(4)
    codes = torch.randint(0, 1024, (30, 8, 16), device="cuda")
    act = torch.zeros(30, 8, dtype=torch.bool, device="cuda")
    act[:, :3] = True
    full, part = StreamCodec(tok, 8), StreamCodec(tok, 8)
    assert part.fast and part.buckets == [1, 2, 4, 8]
    a = full.decode_block(codes, act)
    b = part.decode_block(codes, act, n_active=3)
    assert (a[:, :3] - b[:, :3]).abs().max().item() < 1e-5
