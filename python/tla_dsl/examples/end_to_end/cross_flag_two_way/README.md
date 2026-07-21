# Two-Way Mode-4 Cross-Flag Handshake

This example runs one AIC and both AIV sub-blocks through a three-phase
handshake:

1. AIC increments `aic_to_aiv` for AIV0 and AIV1.
2. Each AIV consumes only its guarded token, computes one output row, and
   independently increments `aiv_to_aic`.
3. AIC waits for both acknowledgements and increments `aic_to_aiv` a second
   time, releasing both AIVs to store their rows.

The repeated AIC sets are separate counter increments. Mode-4 `aiv_id` selects
physical IDs `base` and `base + 16` on AIC; compiler-generated AIV control flow
restricts each operation to the matching `sub_block_idx`.

Build without a device launch:

```bash
python cross_flag_two_way.py --build-only --force-recompile
```

Run on an Ascend 950 device:

```bash
python cross_flag_two_way.py \
  --run --device 0 --block 1 --force-recompile
```

A passing run prints `aiv0_row_ok=True`, `aiv1_row_ok=True`, and
`out_equals_expected=True`.
