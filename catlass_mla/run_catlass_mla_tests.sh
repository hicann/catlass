#!/bin/bash
set -e

n_heads_list=(8 12 16 25 40 64 128)
bsz_list=(1 2 4 8 16 32 64 128 256 512 1024 2048)
kvSeqlen_list=(128 256 512 1024 2048)

for n_heads in "${n_heads_list[@]}"; do
  for batchSize in "${bsz_list[@]}"; do
    for kvSeqlen in "${kvSeqlen_list[@]}"; do
      numBlock=$(( batchSize * kvSeqlen / 128 ))

      echo "==================================="
      echo "batchSize = $batchSize"
      echo "kvSeqlen  = $kvSeqlen"
      echo "numBlock  = $numBlock"
      echo "n_heads   = $n_heads"
      echo "==================================="

      python catlass_mla.py --test --bsz $batchSize --seqlen $kvSeqlen --n_heads $n_heads
      echo ""
    done
  done
done
