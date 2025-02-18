# cutlass

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