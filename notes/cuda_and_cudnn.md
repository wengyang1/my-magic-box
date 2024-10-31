## cuda install 多CUDA配置
```text
1. 一般来说CUDA安装在 /usr/local/ 目录下，可以在 /usr/local/ 目录下通过ls命令查看已经安装的cuda版本
ls /usr/local
```
```text
下载CUDA：https://developer.nvidia.com/cuda-toolkit-archive
获取下载网址后可以复制浏览器直接下载
安装时可以取消nvidia driver的安装
```
## cudnn
```text
下载cudnn：https://developer.nvidia.com/rdp/cudnn-archive
把include和lib复制到刚才安装的cuda中
# 复制文件
sudo cp cudnn-linux-x86_64-8.4.0.27_cuda11.6-archive/include/* /usr/local/cuda-11.6/include/
sudo cp cudnn-linux-x86_64-8.4.0.27_cuda11.6-archive/lib/* /usr/local/cuda-11.6/lib64/
# 添加读取权限
sudo chmod a+r /usr/local/cuda-11.6/lib64/*
sudo chmod a+r /usr/local/cuda-11.6/include/*
```
## 设置cuda软链接
```text
由于我们可能会有多个cuda版本，所以设置软连接比较方便我们切换cuda的版本
1.查看当前cuda软链接
ll /usr/local
2.换软链接
cd /usr/local
sudo rm -rf cuda
sudo ln -s /usr/local/cuda-11.x /usr/local/cuda
```
## 设置变量
```text
vim ~/.bashrc
export PATH=$PATH:/usr/local/cuda/bin 
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```
## check cuda version
```text
nvcc -V
```