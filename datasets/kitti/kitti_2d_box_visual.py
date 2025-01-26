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
cv2.imshow('image', image)
cv2.waitKey(0)
cv2.destroyAllWindows()