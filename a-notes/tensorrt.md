## tensorrt install
```text
https://developer.nvidia.com/tensorrt/download
选择与你的CUDA版本相匹配的TensorRT版本。

使用tar.gz文件安装（推荐）
大多数情况下，你可能需要下载TensorRT的tar.gz文件并手动安装。
下载TensorRT的tar.gz文件。
解压文件到指定目录。
vim ~/.bashrc
将TensorRT的lib目录添加到环境变量LD_LIBRARY_PATH中。
将TensorRT的bin目录添加到环境变量PATH中。
export PATH=$PATH:xxxx/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:xxxx/lib
```

## python版本的tensorrt安装
```text
如果需要Python绑定，进入TensorRT的python目录，并使用pip安装相应的wheel文件。
```