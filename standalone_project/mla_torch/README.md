Standalone out-of-source MLA example with PyTorch interface

## Usage

```bash
export CATLASS_DIR=/mounted_home/work_code/catlass_fork  # adjust to your main catlass dir; tested with catlass-v1-stable/
ls $CATLASS_DIR  # double check

# this project dir can also live in a standalone repo, separate from main catlass repo
cd python_extension
rm -rf build output torch_catlass.egg-info/
python setup.py bdist_wheel --dist-dir ./output/
pip install --force-reinstall ./output/*.whl

cd ..

# run test
python tests/test_attention.py | tee run_test_mla.log
```
