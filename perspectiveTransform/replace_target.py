import cv2
import numpy as np
from codes.my_util import show_image

src_image = cv2.imread('../images/saber.jpg')
dst_image = cv2.imread('../images/test.jpeg')
# show_image('saber', src_image)
show_image('test', dst_image)
# 目标图像中不规则四边形顶点 todo 一般需要用检测手段获取坐标
dst_x0, dst_y0 = 59, 65
dst_x1, dst_y1 = 289, 107
dst_x2, dst_y2 = 305, 257
dst_x3, dst_y3 = 51, 244
dst_points = np.array([[dst_x0, dst_y0], [dst_x1, dst_y1], [dst_x2, dst_y2], [dst_x3, dst_y3]], dtype=np.float32)
dst_points = dst_points.reshape((-1, 1, 2))
# todo 根据透视变换的目标区域将图片resize到合适大小，否则透视变换后清晰度会大幅降低，经过比较 INTER_AREA 插值法最优
new_width, new_height = dst_x2 - dst_x0, dst_y2 - dst_y0
src_image = cv2.resize(src_image, (new_width, new_height), interpolation=cv2.INTER_AREA)
show_image('src_image', src_image)
x0, y0, x1, y1 = 0, 0, src_image.shape[1], src_image.shape[0]
src_rect = np.float32([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])
# 计算透视变换矩阵
M = cv2.getPerspectiveTransform(src_rect, dst_points)
# 应用透视变换
transformed_image = cv2.warpPerspective(src_image, M, (dst_image.shape[1], dst_image.shape[0]))
show_image('transformed_image', transformed_image)
# 目标图片四边形区域填黑色 todo:直线抗锯齿 cv2.LINE_AA
dst_image_copy = dst_image.copy()
dst_points = dst_points.astype(np.int32)
cv2.fillPoly(dst_image_copy, [dst_points], (0, 0, 0), lineType=cv2.LINE_AA)
show_image('dst_image_copy', dst_image_copy)

res = transformed_image + dst_image_copy
show_image('res', res)
