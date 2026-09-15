# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Standalone NPU regression: python x_attention_boundary_regression.py."""
import argparse

import torch
import torch_npu  # noqa: F401
import torch_catlass


def run_case(shape, dtype, mode, device, diagnose=False):
    batch, beam, seq, heads, kv_heads, capacity, decode = shape
    torch.manual_seed(1234)

    def rand(*dims):
        return (torch.rand(dims) - 0.5).to(dtype)

    q = rand(batch, beam, heads, 128)
    sk, sv = [rand(batch, seq, kv_heads, 128) for _ in range(2)]
    uk, uv = [rand(batch, beam, kv_heads, capacity, 128) for _ in range(2)]
    group = heads // kv_heads

    def reference():
        outputs = []
        for b in range(batch):
            k = torch.cat((
                sk[b].permute(1, 0, 2).unsqueeze(0).expand(beam, -1, -1, -1),
                uk[b, :, :, :decode]), dim=2).float().repeat_interleave(group, dim=1)
            v = torch.cat((
                sv[b].permute(1, 0, 2).unsqueeze(0).expand(beam, -1, -1, -1),
                uv[b, :, :, :decode]), dim=2).float().repeat_interleave(group, dim=1)
            scores = (q[b].float().unsqueeze(-2) @ k.transpose(-1, -2)) / (128 ** 0.5)
            outputs.append((scores.softmax(-1) @ v).squeeze(-2))
        return torch.stack(outputs).reshape(batch * beam, heads, 128)

    def invoke():
        blocks = (seq + 127) // 128
        if mode == 0:
            shared = []
            for x in (sk, sv):
                padded = torch.zeros(batch, blocks * 128, kv_heads, 128, dtype=dtype)
                padded[:, :seq] = x
                shared.append(padded.reshape(batch * blocks, 128, kv_heads, 128).to(device))
            unshared = [x.reshape(batch * beam, kv_heads, capacity, 128).to(device) for x in (uk, uv)]
            shared_table = torch.arange(batch * blocks, dtype=torch.int32, device=device).reshape(batch, blocks)
            unshared_table = None
        else:
            shared = [x.reshape(batch * seq, kv_heads, 128).to(device) for x in (sk, sv)]
            unshared = [x.to(device) for x in (uk, uv)]
            shared_table = None
            unshared_table = torch.arange(batch, dtype=torch.int32, device=device)
        return torch_catlass.x_attention(
            q.reshape(batch * beam, heads, 128).to(device), *shared, *unshared,
            unshared_table,
            torch.full((batch,), seq, dtype=torch.int32, device=device),
            torch.tensor([decode], dtype=torch.int32, device=device), shared_table,
        ).cpu().float()

    tol = 0.00195 if dtype == torch.float16 else 0.0156
    failures = []

    def check(stage, output, expected, rtol=tol):
        finite = torch.isfinite(output)
        close = torch.isclose(output, expected, atol=tol, rtol=rtol)
        bad_rows = (~close).any(-1).nonzero()[:6].tolist()
        values = output[finite]
        print(f"STAGE={stage} finite={int(finite.sum())}/{output.numel()} "
              f"close={close.float().mean().item():.6f} "
              f"range={([values.min().item(), values.max().item()] if values.numel() else [])} "
              f"bad_rows={bad_rows}", flush=True)
        for row in bad_rows[:2]:
            i, j = row
            print(f"  row={row} actual={output[i,j,:4].tolist()} "
                  f"expected={expected[i,j,:4].tolist()}", flush=True)
        try:
            torch.testing.assert_close(output, expected, atol=tol, rtol=rtol)
        except AssertionError:
            if not diagnose:
                raise
            failures.append(stage)

    expected = reference()
    actual = invoke()
    check("random", actual, expected)
    if diagnose:
        check("same_input_repeat", invoke(), expected)

    # Invalid cache slots must not affect the result, even after repeated launches.
    if decode < capacity:
        uk[:, :, :, decode:] = 5
        uv[:, :, :, decode:] = -5
        for repeat in range(3):
            # Compare to the CPU golden: a bad first NPU result must not become the reference.
            check(f"unused_slots_changed_{repeat}", invoke(), expected)

    # With zero Q and V=1, attention is exactly one regardless of cache partition.
    q.zero_()
    sv.fill_(1)
    uv.fill_(1)
    check("zero_q_all_v_one", invoke(), torch.ones_like(expected), rtol=0)
    if diagnose:
        uv.zero_()
        check("zero_q_shared_v_one", invoke(), torch.full_like(expected, seq / (seq + decode)), rtol=0)
        sv.zero_()
        uv.fill_(1)
        check("zero_q_unshared_v_one", invoke(), torch.full_like(expected, decode / (seq + decode)), rtol=0)
    if failures:
        raise AssertionError(f"FAILED_STAGES={failures}")
    print(f"PASS shape={shape} dtype={dtype} mode={mode}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--diagnose", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.npu.set_device(args.device)
    device = f"npu:{args.device}"
    shapes = [
        (1, 9, 40, 45, 15, 1, 1),       # odd GQA / incomplete 32-byte P block
        (1, 1, 448, 306, 17, 3, 3),     # shared M=1
        (4, 40, 40, 128, 1, 3, 1),      # GQA=128 / partially filled cache
        (1, 2, 40, 128, 1, 256, 1),     # maximum unshared tile / UB capacity
        (1, 1, 33, 1, 1, 1, 1),        # empty vector sub-core / single row
    ]
    failures = []
    for shape in shapes:
        for dtype in (torch.float16, torch.bfloat16):
            for mode in (0, 1):
                print(f"CASE shape={shape} dtype={dtype} mode={mode}", flush=True)
                try:
                    run_case(shape, dtype, mode, device, diagnose=args.diagnose)
                except AssertionError as error:
                    if not args.diagnose:
                        raise
                    failures.append((shape, str(dtype), mode, str(error)))
                    print(error, flush=True)
    if failures:
        raise SystemExit(f"FAILED_CASES={len(failures)}: {failures}")
    print("XATTENTION_BOUNDARY_REGRESSION_OK", flush=True)
