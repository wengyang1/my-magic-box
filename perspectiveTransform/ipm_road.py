import cv2
import numpy as np
from codes.my_util import *

# 读取原始图像
original_image = cv2.imread('../images/road.png')
show_image('original_image', original_image)
# 定义透视变换所需的点
# 原始图像上的点（注意：这里的坐标应该根据图像的实际内容来调整）
image_points = np.array([
    (10, 457),
    (467, 262),
    (557, 262),
    (1000, 457)
], np.int32)
original_image_red = original_image.copy()
for point in image_points:
    cv2.circle(original_image_red, point, 2, (0, 0, 255), -1)
show_image('original_image_red', original_image_red)

# 目标图像上的点（同样，这里的坐标也应该根据需求来调整）
x0, y0, x1, y1 = int(original_image.shape[1] / 8 * 3), original_image.shape[0], int(original_image.shape[1] / 8 * 5), 0
objective_points = np.array([
    (x0, y0),
    (x0, y1),
    (x1, y1),
    (x1, y0),
], np.float32)
# todo转换数据类型，因为cv2.getPerspectiveTransform需要这种格式
image_points = image_points.astype(np.float32).reshape(-1, 2)
objective_points = objective_points.reshape(-1, 2)

# 计算透视变换矩阵，todo需要注意 getPerspectiveTransform 接收的参数类型
transform = cv2.getPerspectiveTransform(image_points, objective_points)

# 应用透视变换
perspective_image = cv2.warpPerspective(original_image, transform,
                                        (original_image.shape[1], original_image.shape[0]))

show_image('perspective_image', perspective_image)
# sharpen_color_image(perspective_image)
