## 成熟的视觉方案
```text
https://news.sohu.com/a/772060715_121124477
https://blog.csdn.net/weixin_45625942/article/details/129421699
https://www.zhihu.com/question/614057120/answer/3256409271
bevformer+occupancy是一种主流 纯视觉方案
https://www.cvlibs.net/publications/Mescheder2019CVPR.pdf
https://github.com/autonomousvision/occupancy_networks
=======彻底搞清楚车载相机以及视觉方案的细节
https://blog.csdn.net/qq_41920323/article/details/142149832?ops_request_misc=&request_id=&biz_id=102&utm_term=bev%E4%BD%BF%E7%94%A8%E7%9A%84%E7%9B%B8%E6%9C%BA&utm_medium=distribute.pc_search_result.none-task-blog-2~all~sobaiduweb~default-7-142149832.nonecase&spm=1018.2226.3001.4187
https://blog.csdn.net/lovely_yoshino/article/details/120900854?ops_request_misc=&request_id=&biz_id=102&utm_term=IPM&utm_medium=distribute.pc_search_result.none-task-blog-2~all~sobaiduweb~default-2-120900854.nonecase&spm=1018.2226.3001.4187
```
```text
BEV，BirdEye View，即鸟瞰图，通过6V前侧后视，或4V环视，将图像拼接成从上向下的俯视图，
从而可以非常直观的看到汽车周围的环境。当然激光雷达点云同样可以表示为BEV，而且点云处理的操作会更方便。
现在很多车上已经搭载了BEV功能，可能叫做360度全景影像，或AVM（around view monitoring），
全景影像监测，用来倒车时观察周围环境。
BEV主要步骤可以简单概括为标定、去畸变和拼接，如果是3D情况还会将BEV图像投影到一个碗状的模型上，这样就可以拖动观察。
IPM，Inverse Perspective Mapping，逆透视变换，在视觉BEV的生成中起到了重要的作用，在前视摄像头拍摄的图像中，
由于透视效应的存在，本来平行的事物，在图像中确实相交的，比如平行的车道线变成了相交的。而IPM变换就是消除这种透视效应，
所以也叫逆透视。
比如4V环视（一般都是用4V鱼眼来做BEV图像）生成BEV图像，最终的视角是在汽车正上方，俯视汽车，
这个图像对应了一个虚拟的相机，它也有内参和外参，这些参数来自IPM过程。
鱼眼和针孔(pinhole)的区别在于，鱼眼相机，即fisheye camera的fov(field of view)更大，
同时其代价就是畸变会明显更严重。鱼眼为了实现更大的视角（一般会大于140度），使用了一系列镜片的组合，
这也导致它无法用传统的鱼眼相机模型来建模，Kannala-Brandt模型是一种常用的鱼眼镜头模型。
matlab和opencv使用的鱼眼模型具体是什么样的我暂时不清楚，不过matlab有模型转换函数，
可以将matlab的鱼眼模型转换到opencv支持的鱼眼模型（cameraIntrinsicsToOpenCV）。
fisheye camera同样可以使用matlab工具箱来进行标定。
```