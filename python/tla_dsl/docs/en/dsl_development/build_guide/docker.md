---
nav_order: 30
---

# Building a Development Environment with Docker

The development environment built with Docker includes all dependencies required for development and execution, but does not include the CATLASS source code. Mount it from the host at startup, or clone it yourself.

## Building the Image

```bash
# Replace /path/to/catlass with the actual path of the CATLASS repository you cloned
cd /path/to/catlass/python/tla_dsl
# In mainland China, append --default-mirror to speed things up with a mirror (drop it if the mirror is unreachable):
#   bash build_docker_image.sh swr.cn-south-1.myhuaweicloud.com/ascendhub/cann:9.1.0-950-ubuntu22.04-py3.12 --default-mirror
bash build_docker_image.sh swr.cn-south-1.myhuaweicloud.com/ascendhub/cann:9.1.0-950-ubuntu22.04-py3.12
```

The base image name is only an example. Choose a CANN image matching your device, driver, and Python version from [AscendHub](https://www.hiascend.com/developer/ascendhub/detail/17da20d1c2b6493cb38765adeba85884), and pass the full name as the first argument. The output image is `ascend-catlass-dsl:<base image tag>`.

If you are in mainland China, appending `--default-mirror` to use a mirror is recommended. Read the script contents to learn more about the image configuration.

```bash
bash build_docker_image.sh swr.cn-south-1.myhuaweicloud.com/ascendhub/cann:9.1.0-950-ubuntu22.04-py3.12 --default-mirror
```

Building AscendNPU-IR is time-consuming and resource-intensive. When memory or disk I/O is limited, reduce the concurrency with `--build-jobs`.

## Starting the Container

```bash
docker run                          \
    --rm                            \
    --name ascend-catlass-dsl-dev   \
    --device /dev/davinci0          \
    --device /dev/davinci_manager   \
    --device /dev/hisi_hdc          \
    -v /usr/local/dcmi:/usr/local/dcmi                                              \
    -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi                                \
    -v /usr/local/Ascend/driver/lib64/:/usr/local/Ascend/driver/lib64/              \
    -v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info  \
    -v /etc/ascend_install.info:/etc/ascend_install.info                            \
    -v /path/to/catlass:/workspace/catlass                                          \
    -w /workspace/catlass/python/tla_dsl                                            \
    -it ascend-catlass-dsl:9.1.0-950-ubuntu22.04-py3.12 bash
```

If you are using a supernode device based on the Ascend950 series chips, such as [Atlas 950 SuperPoD liquid-cooled supernode](https://www.hiascend.com/hardware/cluster?tag=950) or [Atlas 850E air-cooled supernode](https://www.hiascend.com/hardware/cluster?tag=850e), you may need to mount additional devices related to the Lingqu bus:

```diff
docker run                          \
    --rm                            \
    --name ascend-catlass-dsl-dev   \
    --device /dev/davinci0          \
    --device /dev/davinci_manager   \
    --device /dev/hisi_hdc          \
+   --device /dev/ubcore            \
+   --device /dev/uburma            \
+   --device /dev/ummu              \
    -v /usr/local/dcmi:/usr/local/dcmi                                              \
    -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi                                \
    -v /usr/local/Ascend/driver/lib64/:/usr/local/Ascend/driver/lib64/              \
    -v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info  \
    -v /etc/ascend_install.info:/etc/ascend_install.info                            \
    -v /path/to/catlass:/workspace/catlass                                          \
    -w /workspace/catlass/python/tla_dsl                                            \
    -it ascend-catlass-dsl:9.1.0-950-ubuntu22.04-py3.12 bash
```

Driver paths are based on the actual installation paths on the host; inside the container you can check the devices with `npu-smi info`.
