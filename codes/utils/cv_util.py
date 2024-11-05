import cv2
import numpy as pd


def show_image(win_name, one_mat):
    cv2.imshow(win_name, one_mat)
    cv2.waitKey(0)

