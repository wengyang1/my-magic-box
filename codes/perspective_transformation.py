import cv2
import numpy as np

# 读取图像
image = cv2.imread('../data/example.gif')
cv2.imshow('test',image)
cv2.waitKey(0)
# 定义源图像中的四个顶点（不规则四边形，按顺时针顺序）
# 这些点可以通过手动选择、自动检测（如边缘检测、轮廓检测）等方法获得
# 在这里，我们假设已经知道了这些点的坐标
pts_src = np.float32([[100, 150], [400, 100], [350, 400], [50, 350]])

# 定义目标图像中的四个顶点（规则矩形，按顺时针顺序）
# 这些点定义了变换后的矩形区域
width, height = 300, 400  # 目标矩形的宽度和高度
pts_dst = np.float32([[0, 0], [width, 0], [width, height], [0, height]])

# 计算透视变换矩阵
M = cv2.getPerspectiveTransform(pts_src, pts_dst)

# 应用透视变换
# 注意：这里我们使用cv2.warpPerspective的第三个参数为目标图像的尺寸
warped_image = cv2.warpPerspective(image, M, (width, height))

# 显示结果
cv2.imshow('Original Image with Points', cv2.drawContours(image.copy(), [pts_src.astype(np.int32)], -1, (0, 255, 0), 2))
cv2.imshow('Warped Image', warped_image)

# 保存结果图像（可选）

# 等待按键按下，然后关闭所有窗口
cv2.waitKey(0)
cv2.destroyAllWindows()