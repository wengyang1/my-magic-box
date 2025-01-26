## mmlab是什么
```text
MMLab，即多媒体实验室（Multimedia Laboratory），通常指的是由港中文的汤晓鸥教授建立的实验室及其在GitHub上开源的一系列算法。以下是对MMLab的详细介绍：

一、MMLab实验室
MMLab实验室在多媒体和计算机视觉领域有着深厚的研究背景。实验室的研究方向广泛，涵盖了机器学习、强化学习、半监督/弱监督/自监督学习等前沿方法和理论，同时也涉及长视频理解、3D视觉、生成模型等计算机视觉新兴方向，以及物体检测、动作识别等核心方向的性能突破。

二、MMLab算法库
MMLab算法库是MMLab实验室在GitHub上开源的一系列算法，主要包括目标检测、3D点云、图像分割、图像识别、视频理解，以及AIGC等相关应用。此外，算法库还包括了部署工具MMDeploy，用于将OpenMMLab旗下的开源模型进行部署。MMLab算法库涵盖了计算机视觉相关的几乎所有领域，被学术界和工业界广泛使用。

具体开源工具库及其特点如下：

MMCV：用于计算机视觉研究的基础Python库，支持OpenMMLab旗下其他开源库。主要功能是I/O、图像视频处理、标注可视化、各种CNN架构、各类CUDA操作算子。
MMDetection：基于PyTorch的开源目标检测工具箱，是OpenMMLab最知名的开源库之一。支持开箱即用的多模态/单模态检测器，以及室内/室外检测器。
MMClassification：基于PyTorch的开源图像分类工具箱。提供各种骨干与预训练模型、Bag of training tricks、大规模训练配置等，具有高效率与可扩展性。
MMPose：基于PyTorch的开源姿势估计工具箱。
MMAction：基于PyTorch开放源代码的工具箱，用于动作理解。
MMAction2：是MMAction的升级版，模块化设计，支持多种数据集和多重动作理解框架，完善的测试和记录。比MMAction支持的算法更多，速度更快。
mmOCR：用于字符识别的开源工具箱。
mmSegmentation：用于图像分割的开源工具箱。
三、MMLab的应用与影响
MMLab的开源算法库和工具为计算机视觉领域的研究者和开发者提供了丰富的资源和便利。这些算法库和工具不仅提高了研究效率，还推动了计算机视觉技术的快速发展。同时，MMLab的研究成果也在学术界和工业界产生了广泛的影响。

综上所述，MMLab是一个在计算机视觉领域具有深厚研究背景和广泛影响力的实验室。其开源的算法库和工具为计算机视觉技术的研究和发展提供了重要的支持。
```
## mmcv下载
```text
https://download.openmmlab.com/mmcv/dist/cu113/torch1.11.0/index.html
https://download.openmmlab.com/mmcv/dist/cpu/torch1.11.0/index.html
注意：需要匹配cuda版本、torch版本，根据你环境中的pytorch版本来下载对应的mmcv库
```
## mmcv项目详解
### mmpretrain
```text
https://github.com/open-mmlab/mmpretrain
OpenMMLab Pre-training Toolbox and Benchmark
```