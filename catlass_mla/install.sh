cd python_extension
rm -rf build output torch_catlass.egg-info/
python setup.py bdist_wheel --dist-dir ./output/
pip install --force-reinstall ./output/*.whl

cd ..

# for tests (bfloat16 reference calculation)
pip install ml_dtypes
