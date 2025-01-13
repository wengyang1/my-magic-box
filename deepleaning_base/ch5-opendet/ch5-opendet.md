## error

```text
python setup.py develop

IndexError: list index out of range

是nvidia没安装好, install nvidia driver
nvidia-smi
```


```text
cd tools
 
python demo.py --cfg_file cfgs/kitti_models/pointpillar.yaml \
    --ckpt demo-KITTI/pointpillar_7728.pth \
    --data_path demo-KITTI/000000.bin 

RuntimeError: 
cannot statically infer the expected size of a list in this context:
  File "/home/wd-racing/OpenPCDet/venv/lib/python3.8/site-packages/kornia/geometry/conversions.py", line 553
出现上面的错误，可很可能是版本不对，卸载后安装其他版本
pip install kornia==0.6.5

```