import cv2
import numpy as np
from utils.cv_util import show_image

src_image = cv2.imread('../images/saber.jpg')
dst_image = cv2.imread('../images/test.jpeg')
# show_image('saber', src_image)
# show_image('test', dst_image)
# 目标图像中不规则四边形顶点
dst_x0, dst_y0 = 48, 47
dst_x1, dst_y1 = 300, 93
dst_x2, dst_y2 = 321, 266
dst_x3, dst_y3 = 35, 253
dst_points = np.array([[dst_x0, dst_y0], [dst_x1, dst_y1], [dst_x2, dst_y2], [dst_x3, dst_y3]], dtype=np.float32)
dst_points = dst_points.reshape((-1, 1, 2))
# 根据透视变换的目标区域将图片resize到合适大小，否则透视变换后清晰度会大幅降低，经过比较INTER_AREA插值法最优
new_width, new_height = dst_x2 - dst_x0, dst_y2 - dst_y0
src_image = cv2.resize(src_image, (new_width, new_height), interpolation=cv2.INTER_AREA)
show_image('src_image', src_image)
x0, y0, x1, y1 = 0, 0, src_image.shape[1], src_image.shape[0]
src_rect = np.float32([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])
# 计算透视变换矩阵
M = cv2.getPerspectiveTransform(src_rect, dst_points)
transformed_image = cv2.warpPerspective(src_image, M, (
    int(np.linalg.norm(dst_points[0] - dst_points[2])), int(np.linalg.norm(dst_points[1] - dst_points[3]))))
show_image('transformed_image', transformed_image)
# 目标区域填充黑色
dst_image_copy = dst_image.copy()
dst_points = dst_points.astype(np.int32)
cv2.fillPoly(dst_image_copy, [dst_points], (0, 0, 0), lineType=cv2.LINE_8)
show_image('dst_image_copy', dst_image_copy)

# 转换图片加embedding
transformed_image_embedded = np.zeros(dst_image.shape[:], dtype=np.uint8)
transformed_image_embedded[0:transformed_image.shape[0], 0:transformed_image.shape[1], :] = transformed_image
show_image('transformed_image_embedded', transformed_image_embedded)

res = transformed_image_embedded + dst_image_copy
show_image('res', res)
