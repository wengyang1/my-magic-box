import cv2
import numpy as np
from my_util import *




# 使用示例
folder_path = './'
result_image = concatenate_images_in_folder(folder_path)
cv2.imwrite('../perspectiveTransform/res.jpg', result_image)
# 或者使用cv2.imshow('Concatenated Image', result_image)显示图片，然后按任意键关闭窗口
# cv2.waitKey(0)
# cv2.destroyAllWindows()