import cv2
import numpy as np
import os


def show_image(win_name, one_mat):
    cv2.imshow(win_name, one_mat)
    cv2.waitKey(0)


def enhance_image_clarity(image):
    # 转换为灰度图像（可选，但通常锐化在灰度图像上效果更明显）
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 应用高斯模糊来降噪（可选，但通常有助于减少锐化过程中的噪声放大）
    blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)

    # 使用拉普拉斯算子进行锐化
    laplacian = cv2.Laplacian(blurred_image, cv2.CV_64F)

    # 将拉普拉斯锐化结果转换回uint8类型（因为OpenCV函数通常要求输入和输出图像具有相同的类型）
    laplacian_uint8 = cv2.convertScaleAbs(laplacian)

    # 如果需要处理彩色图像，可以只对每个通道应用相同的锐化过程，然后合并通道
    # 但在这个例子中，我们简化处理并只处理灰度图像
    show_image('output', laplacian_uint8)


def sharpen_color_image(image):
    # 转换到YUV颜色空间
    yuv_image = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)

    # 分离Y、U、V通道
    y_channel, u_channel, v_channel = cv2.split(yuv_image)
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])  # 拉普拉斯核

    # kernel = np.array([[0, 0, 0], [0, 5, 0], [0, 0, 0]])  # 拉普拉斯核
    # 对Y通道应用拉普拉斯锐化
    sharpened_y = cv2.filter2D(y_channel, -1, kernel)

    # 合并锐化后的Y通道和原始的U、V通道
    sharpened_yuv_image = cv2.merge([sharpened_y, u_channel, v_channel])

    # 转换回BGR颜色空间
    sharpened_image = cv2.cvtColor(sharpened_yuv_image, cv2.COLOR_YUV2BGR)

    show_image('sharpened_image', sharpened_image)

def concatenate_images_in_folder(folder_path):
    # 获取文件夹中所有图片文件的路径
    image_files = [os.path.join(folder_path, f) for f in sorted(os.listdir(folder_path)) if
                   f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]

    if not image_files:
        raise ValueError("No images found in the folder.")

    # 读取第一张图片并获取其宽度
    img1 = cv2.imread(image_files[0])
    width_top = img1.shape[1]
    total_height = img1.shape[0]

    # 创建一个列表来存储所有调整大小后的图片
    resized_images = [img1]

    # 遍历剩余的图片并调整大小
    for img_path in image_files[1:]:
        img = cv2.imread(img_path)
        height, width = img.shape[:2]
        scale_factor = width_top / width
        new_width = width_top
        new_height = int(height * scale_factor)
        img_resized = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)
        resized_images.append(img_resized)
        total_height += new_height

    # 创建一个新的空白图片用于拼接
    concatenated_image = np.zeros((total_height, width_top, 3), dtype=np.uint8)
    concatenated_image.fill(255)  # 可选：填充为白色背景，如果需要黑色背景则使用0

    # 将所有图片复制到新的空白图片中
    y_offset = 0
    for img in resized_images:
        concatenated_image[y_offset:y_offset + img.shape[0], 0:img.shape[1]] = img
        y_offset += img.shape[0]

    return concatenated_image