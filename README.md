# MLA Kernel with Python Interface

This project provides an extended **MLA (Multi-head Linear Attention) kernel** implementation, based on [Ascend Catlass](https://gitee.com/ascend/catlass/tree/master).
It includes Python bindings, benchmark scripts, and unit tests for quick integration and evaluation.

## Features

Compared to the original Catlass MLA kernel, this version introduces:

* **Python Interface**: Simple and direct Python API for experiments and benchmarking.
* **External Memory Management**: Memory buffers are managed externally via a `prepare` function, allowing tighter integration into larger systems.
* **Log-Sum-Exp (LSE) Extraction**: Support for `return_lse=True` option to export LSE values from MLA computations.
* **Configurable Softmax Scale**: Softmax scaling factor can be provided externally for more flexibility.

These extensions make the kernel suitable for research and production settings involving custom attention mechanisms.

---

## Installation

### 1. Environment Setup

Enable the Ascend CANN environment (example for root installation):

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export LD_LIBRARY_PATH=/usr/local/Ascend/driver/lib64/driver:$LD_LIBRARY_PATH
```

If using conda, activate your environment first:

```bash
source $CONDA_HOME/bin/activate
```

### 2. Install Dependencies

Clone the repository and set up Catlass:

```bash
cd ascendc-samples
export CATLASS_DIR=$(pwd)/ref_catlass/catlass
cd catlass_mla
bash install.sh
```

This will build the kernel and install required Python dependencies.

---

## Usage

### Run Tests

```bash
bash run_catlass_mla_tests.sh
```

### Run Benchmarks

```bash
python catlass_mla.py --bench \
                      --n_heads 128
                      --bsz 1 2 4 8 16 32 64 128 256 512 1024 \
                      --seqlen 128 256 512 1024 2048 4096 \
```

---

## 📜 License

This project is open-sourced under the [Apache 2.0 License](LICENSE).

---

## Acknowledgements

This work builds upon [Catlass](https://gitee.com/ascend/catlass) from Huawei Ascend.
We extend their MLA kernel with additional features for research and deployment.

---