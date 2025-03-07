# cutlass

#运行方法
1. 首先，进入example
cd ascendc-operator-templates-gemv/examples/04_gemv_aiv
2. 进入所要测试的数据类型与排布方式目录下，以fp16列优先为例
cd /01_fp16_cm_gemv_aiv
3. 验证正确性可以通过调用make.sh或者run.sh $M $N $deviceId $mode
4. 调用msprof验证可以调用 run_profiling.sh $M $N $deviceId $mode
# 仓库日记
25.01.08

建立华为cutlass项目仓库
# 分支管理
自己在自己的工作目录下，进行相应的工作，在[cutlass项目地址](https://gitee.com/owqowq/cutlass)新建一个分支${BranchName}

之后，在自己的工作目录下执行这些命令
```shell
# 最好新建一个.git文件，因为不知道之前.git里有啥
git init
git remote add origin https://gitee.com/owqowq/cutlass.git
git pull origin master
git branch ${BranchName}
git checkout ${BranchName}
git add ${相应的文件}
git commit -m "${相应的内容}"
git push origin ${BranchName}
```