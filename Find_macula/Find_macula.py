from glob import glob
import numpy as np
import pandas as pd
import os
import cv2
import matplotlib.pyplot as plt

def processing(path, Gap):
    
    orginal_img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    
    adge_img = cv2.Canny(orginal_img,100,300)
    
    result_img = np.zeros_like(adge_img, dtype="uint8")
    
    width = adge_img.shape[1]
    height = adge_img.shape[0]

    # min_row_index = []

    result_row_list = []
    result_col_list = []

    for col in range(width):

        row_index = []

        for row in range(height):

            value = adge_img[row, col]

            # 하나의 col에서 값을 가지고 있는 index 중에 가장 위에 있는 인덱스의 값을 위해서 배열에 넣어주기
            if  value == 255:
                row_index.append(row)
                # print(row_index)    

        if len(row_index) == 0:
            result_img[row, col] = 0

        else:
            min_index = min(row_index)
            # print(min_index)
            result_img[min_index, col] = 255

        # min_row_index.append(min_index)

    for col in range(width):

        for row in range(height):
            if result_img[row, col] == 255:

                result_row_list.append(row)
                result_col_list.append(col)

    # macula_row_index = max(row_list)
    macula_row_index = max(result_row_list)
    # index = row_list.index(max(row_list))
    index = result_row_list.index(macula_row_index)
    macula_col_index = result_col_list[index]
    
    col_crop_standard = macula_col_index
    # print(width)
    start = col_crop_standard - Gap
    end = col_crop_standard + Gap
    cropped_img = orginal_img[:, start : end]
    # width_type = width.type
    
    return cropped_img, col_crop_standard