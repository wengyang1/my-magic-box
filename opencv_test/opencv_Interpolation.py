import cv2
from codes.my_util import show_image

# 读取图像
image = cv2.imread('../images/saber.jpg')

# 设置缩放比例
scale_percent = 20  # 缩小到50%
width = int(image.shape[1] * scale_percent / 100)
height = int(image.shape[0] * scale_percent / 100)
dim = (width, height)

# 使用不同的插值方法进行缩放
# INTER_NEAREST: 最近邻插值
nearest_image = cv2.resize(image, dim, interpolation=cv2.INTER_NEAREST)

# INTER_LINEAR: 双线性插值（默认）
linear_image = cv2.resize(image, dim, interpolation=cv2.INTER_LINEAR)

# INTER_CUBIC: 双三次插值
cubic_image = cv2.resize(image, dim, interpolation=cv2.INTER_CUBIC)

# INTER_LANCZOS4: Lanczos插值
lanczos_image = cv2.resize(image, dim, interpolation=cv2.INTER_LANCZOS4)
area_image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)

# 显示原始图像和缩放后的图像
show_image('Original Image', image)
show_image('Nearest Neighbor Interpolation', nearest_image)
show_image('Bilinear Interpolation', linear_image)
show_image('Bicubic Interpolation', cubic_image)
show_image('Lanczos Interpolation', lanczos_image)
show_image('INTER_AREA', area_image)

cv2.destroyAllWindows()
