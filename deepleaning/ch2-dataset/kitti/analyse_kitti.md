## label字段含义

| 字段        | 字段⻓度 |  单位 |                     含义                     |
|:----------|:----:|----:|:------------------------------------------:|
| Type      |  1   |   - |                    目标类型                    | 
| Truncated |  1   |   - |       ⽬标截断程度：0~1之间的浮点数 表示⽬标距离图像边界的程度       |
| Occluded  |  1   |   - | ⽬标遮挡程度：0~3之间的整数 0：完全可⻅ 1：部分遮挡 2：⼤部分遮挡 3：未知 |
| Alpha     |  1   |  弧度 |              ⽬标观测⻆：[−pi, pi]               |
| Bbox      |  4   |  像素 |          ⽬标2D检测框位置：左上顶点和右下顶点的像素坐标          |
| Dimensions|  3   |   ⽶ |                3D⽬标尺⼨：⾼、宽、⻓                |
| Location  |  3   |   ⽶ |        ⽬标3D框底⾯中⼼坐标：(x,y,z) ，相机坐标系，         |
| Rotation_y|  1   |  弧度 |              ⽬标朝向⻆：[−pi, pi]               |


```text
KITTI数据集是国际上最大的自动驾驶场景下的计算机视觉算法评测数据集，其label文件中的每个字段都承载着特定的含义。以下是KITTI数据集label中每个字段的详细解释：

物体类别：
第一个字段通常是一个字符串，表示检测到的物体类别。可能的值包括'Car', 'Van', 'Truck', 'Pedestrian', 'Person_sitting', 'Cyclist', 'Tram', 'Misc'以及'DontCare'。其中，'DontCare'标签表示该区域没有被标注，这通常是因为目标物体距离激光雷达太远，超出了雷达的扫描范围。
截断程度：
第二个字段是一个浮点数，范围从0到1，表示物体是否被截断。0表示物体没有被截断，1表示物体被完全截断（即物体的一部分超出了图像边界）。
遮挡程度：
第三个字段是一个整数，表示物体被遮挡的程度。可能的值包括0（完全可见）、1（小部分遮挡）、2（大部分遮挡）和3（完全遮挡或未知）。
观察角度（alpha）：
第四个字段是一个浮点数，表示物体的观察角度。这个角度是在相机坐标系下，以相机原点为中心，相机原点到物体中心的连线为半径，将物体绕相机y轴旋转至相机z轴时，物体方向与相机x轴的夹角。范围是从-π到π。
2D边界框：
第五到第八个字段分别表示物体的2D边界框的左上角和右下角的x、y坐标（在图像像素坐标系下）。这四个值（xmin, ymin, xmax, ymax）定义了物体在图像上的位置。
3D尺寸：
第九到第十一个字段分别表示物体的高度、宽度和长度（单位为米）。这三个值（h, w, l）定义了物体的3D尺寸。
3D位置：
第十二到第十四个字段分别表示物体的3D中心在相机坐标系下的x、y、z坐标。这三个值定义了物体在3D空间中的位置。
3D空间方向（rotation_y）：
第十五个字段是一个浮点数，表示物体的3D空间方向。这是在相机坐标系下，物体的全局方向角（物体前进方向与相机坐标系x轴的夹角），范围也是从-π到π。与alpha不同的是，rotation_y直接在相机坐标中给出，而alpha还会考虑从相机中心到物体中心的矢量来计算物体相对于相机的相对方向。
检测的置信度：
第十六个字段是一个浮点数，表示检测的置信度。这个值通常用于评估检测结果的可靠性。
综上所述，KITTI数据集label中的每个字段都提供了关于检测到的物体的详细信息，这些信息对于自动驾驶系统中的物体检测、跟踪和识别等任务至关重要。
```

### 字段type - kitti类别
```python
import os

def get_first_column_unique_list(folder_path):
    unique_list = set()  # 使用集合来自动去重

    # 遍历文件夹中的所有文件
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.txt'):
                file_path = os.path.join(root, file)
                with open(file_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        # 用空格分割每行
                        columns = line.strip().split()
                        if columns:  # 确保行不为空
                            first_column = columns[0]
                            unique_list.add(first_column)

    return list(unique_list)  # 将集合转换为列表

# 使用示例
folder_path = 'training/label_2'  # 替换为你的文件夹路径
unique_first_columns = get_first_column_unique_list(folder_path)
print(unique_first_columns)
# ['Van', 'Misc', 'Pedestrian', 'Truck', 'Car', 'Cyclist', 'Tram', 'Person_sitting', 'DontCare']
# DontCare 一般是目标太远了，除了bbox其他参数都是默认的，例如
# DontCare -1 -1 -10 800.38 163.67 825.45 184.07 -1 -1 -1 -1000 -1000 -1000 -10
```

### 字段 Truncated
```text
以  [000008.txt]
Car 0.88 3 -0.69 0.00 192.37 402.31 374.00 1.60 1.57 3.23 -2.70 1.74 3.68 -1.29
为例:
0.88 表示被图片边界截断比例0.88
0 表示离图片边界很远，没有被截断
```
### 字段 Occluded
```text
⽬标遮挡程度：0~3之间的整数 0：完全可⻅ 1：部分遮挡 2：⼤部分遮挡 3：未知
这个字段很好理解，字面意思即可
```
### 字段 bbox
```python
words_all = []  # 用于存储所有单词的列表

with open('training/label_2/000008.txt', 'r', encoding='utf-8') as file:
    lines = file.readlines()

for line in lines:  # 直接迭代文件对象，它会按行返回内容
    words_in_line = line.strip().split()  # 去除行尾的换行符和空格，然后按空格分割
    words_all.append(words_in_line)  # 将分割后的单词添加到总列表中

import cv2

image = cv2.imread('training/image_2/000008.png')

for i, words in enumerate(words_all):
    type, x1, y1, x2, y2 = words[0], words[4], words[5], words[6], words[7]
    x1, y1, x2, y2 = int(float(x1)), int(float(y1)), int(float(x2)), int(float(y2))
    print(type, x1, y1, x2, y2)
    color = (0, 255, 0)
    thickness = 2
    cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
    if type == 'DontCare':
        cv2.putText(image, '1', (x1 + 5, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, color, thickness)
cv2.imshow('image', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

### 3d目标框的可视化
```text
https://blog.csdn.net/qq_41204464/article/details/132776800?ops_request_misc=&request_id=&biz_id=102&utm_term=kitti%E5%A6%82%E4%BD%95%E5%9C%A8%E5%9B%BE%E7%89%87%E4%B8%8A%E7%94%BB3d%E6%A1%86&utm_medium=distribute.pc_search_result.none-task-blog-2~all~sobaiduweb~default-1-132776800.142^v101^pc_search_result_base2&spm=1018.2226.3001.4187
```