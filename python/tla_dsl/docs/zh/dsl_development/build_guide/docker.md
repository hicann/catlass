---
nav_order: 30
---

# 使用 Docker 构建开发环境

使用Docker构建的开发环境，包含全部开发运行所需的依赖，但不包含 CATLASS 源码，启动时需从宿主机挂载，或自行克隆。

## 构建镜像

```bash
# /path/to/catlass 需替换为你 clone 的 CATLASS 仓库实际路径
cd /path/to/catlass/python/tla_dsl
# 国内环境推荐追加 --default-mirror 使用镜像源加速（镜像源不可达时去掉）：
#   bash build_docker_image.sh swr.cn-south-1.myhuaweicloud.com/ascendhub/cann:9.1.0-950-ubuntu22.04-py3.12 --default-mirror
bash build_docker_image.sh swr.cn-south-1.myhuaweicloud.com/ascendhub/cann:9.1.0-950-ubuntu22.04-py3.12
```

基础镜像名称只是示例。请从 [AscendHub](https://www.hiascend.com/developer/ascendhub/detail/17da20d1c2b6493cb38765adeba85884) 选择与设备、驱动和 Python 版本匹配的 CANN 镜像，并将完整名称作为第一个参数。输出镜像为 `ascend-catlass-dsl:<基础镜像 tag>`。

若您处于中国大陆，建议追加 `--default-mirror` 使用镜像源加速。可阅读脚本内容，更具体地了解镜像配置。

```bash
bash build_docker_image.sh swr.cn-south-1.myhuaweicloud.com/ascendhub/cann:9.1.0-950-ubuntu22.04-py3.12 --default-mirror
```

AscendNPU-IR 构建耗时和资源占用较高。内存或磁盘 I/O 有限时，通过 `--build-jobs` 降低并发数。

## 启动容器

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

若您使用的是基于Ascend950系列芯片的超节点设备，例如[Atlas 950 SuperPoD 液冷超节点](https://www.hiascend.com/hardware/cluster?tag=950)或[Atlas 850E 风冷超节点](https://www.hiascend.com/hardware/cluster?tag=850e)，可能需要额外挂载灵渠总线相关的设备：

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

驱动路径以宿主机的实际安装路径为准；容器内可通过 `npu-smi info` 检查设备。
