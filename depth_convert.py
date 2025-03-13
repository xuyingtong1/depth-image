# 开发时间：10:4410:44

# 点云深度---→深度图
import os

import cv2
import scipy
import skimage
import numpy as np
from matplotlib import cm
from pypardiso import spsolve
from PIL import Image
import matplotlib.pyplot as plt
from skimage import color


def fill_depth_colorization(imgRgb=None, imgDepthInput=None, alpha=1):
    imgIsNoise = imgDepthInput == 0
    maxImgAbsDepth = np.max(imgDepthInput)
    imgDepth = imgDepthInput / maxImgAbsDepth
    imgDepth[imgDepth > 1] = 1
    (H, W) = imgDepth.shape
    numPix = H * W
    indsM = np.arange(numPix).reshape((W, H)).transpose()
    knownValMask = (imgIsNoise == False).astype(int)
    grayImg = skimage.color.rgb2gray(imgRgb)
    winRad = 1
    len_ = 0
    absImgNdx = 0
    len_window = (2 * winRad + 1) ** 2
    len_zeros = numPix * len_window

    cols = np.zeros(len_zeros) - 1
    rows = np.zeros(len_zeros) - 1
    vals = np.zeros(len_zeros) - 1
    gvals = np.zeros(len_window) - 1

    for j in range(W):
        for i in range(H):
            nWin = 0
            for ii in range(max(0, i - winRad), min(i + winRad + 1, H)):
                for jj in range(max(0, j - winRad), min(j + winRad + 1, W)):
                    if ii == i and jj == j:
                        continue

                    rows[len_] = absImgNdx
                    cols[len_] = indsM[ii, jj]
                    gvals[nWin] = grayImg[ii, jj]

                    len_ = len_ + 1
                    nWin = nWin + 1

            curVal = grayImg[i, j]
            gvals[nWin] = curVal
            c_var = np.mean((gvals[:nWin + 1] - np.mean(gvals[:nWin + 1])) ** 2)

            csig = c_var * 0.6
            mgv = np.min((gvals[:nWin] - curVal) ** 2)
            if csig < -mgv / np.log(0.01):
                csig = -mgv / np.log(0.01)

            if csig < 2e-06:
                csig = 2e-06

            gvals[:nWin] = np.exp(-(gvals[:nWin] - curVal) ** 2 / csig)
            gvals[:nWin] = gvals[:nWin] / sum(gvals[:nWin])
            vals[len_ - nWin:len_] = -gvals[:nWin]

            # Now the self-reference (along the diagonal).
            rows[len_] = absImgNdx
            cols[len_] = absImgNdx
            vals[len_] = 1  # sum(gvals(1:nWin))

            len_ = len_ + 1
            absImgNdx = absImgNdx + 1

    vals = vals[:len_]
    cols = cols[:len_]
    rows = rows[:len_]
    A = scipy.sparse.csr_matrix((vals, (rows, cols)), (numPix, numPix))

    rows = np.arange(0, numPix)
    cols = np.arange(0, numPix)
    vals = (knownValMask * alpha).transpose().reshape(numPix)
    G = scipy.sparse.csr_matrix((vals, (rows, cols)), (numPix, numPix))

    A = A + G
    b = np.multiply(vals.reshape(numPix), imgDepth.flatten('F'))

    # print ('Solving system..')

    new_vals =  (A, b)
    new_vals = np.reshape(new_vals, (H, W), 'F')

    # print ('Done.')

    denoisedDepthImg = new_vals * maxImgAbsDepth

    output = denoisedDepthImg.reshape((H, W)).astype('float32')

    output = np.multiply(output, (1 - knownValMask)) + imgDepthInput

    return output




if __name__ == "__main__":

    rgb_image_path = r"imgs/0000000005.png"  # RGB图像路径
    depth_image_path = r"depths/0000000005.png"  # 深度图像路径

    imgRgb = cv2.imread(rgb_image_path)
    imgRgb = cv2.cvtColor(imgRgb, cv2.COLOR_BGR2RGB)
    imgRgb = imgRgb / 255.0

    imgDepthInput = cv2.imread(depth_image_path, cv2.IMREAD_GRAYSCALE)
    # imgDepthInput = imgDepthInput / imgDepthInput.max()
    imgDepthInput = imgDepthInput.astype(np.float32)

    imgDepthOutput = fill_depth_colorization(imgRgb = imgRgb, imgDepthInput = imgDepthInput)

    plt.figure(figsize=(12, 6))

    plt.subplot(1, 3, 1)
    plt.imshow(imgRgb)
    plt.title('Original RGB Image')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(imgDepthInput, cmap='gray')
    plt.title('Original Depth Image')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(imgDepthOutput, cmap='plasma_r')
    plt.title('Processed Depth Image')
    plt.axis('off')

    plt.tight_layout()
    plt.show()

# if __name__ == "__main__":
#     # 设置文件夹路径
#     rgb_folder = r"imgs"
#     depth_folder = r"depths"
#     output_folder = r"outputs"
#
#     if not os.path.exists(output_folder):
#         os.makedirs(output_folder)
#
#     rgb_files = sorted(os.listdir(rgb_folder))
#     depth_files = sorted(os.listdir(depth_folder))
#
#     assert len(rgb_files) == len(depth_files), "RGB和深度图像数量不匹配"
#
#     for rgb_file, depth_file in zip(rgb_files, depth_files):
#         rgb_path = os.path.join(rgb_folder, rgb_file)
#         depth_path = os.path.join(depth_folder, depth_file)
#
#         imgRgb = cv2.imread(rgb_path)
#         imgRgb = cv2.cvtColor(imgRgb, cv2.COLOR_BGR2RGB)
#         imgRgb = imgRgb / 255.0
#
#         imgDepthInput = cv2.imread(depth_path, cv2.IMREAD_GRAYSCALE)
#         imgDepthInput = imgDepthInput.astype(np.float32)
#
#         imgDepthOutput = fill_depth_colorization(imgRgb=imgRgb, imgDepthInput=imgDepthInput, alpha=1)
#
#         imgDepthOutput_normalized = (imgDepthOutput - np.min(imgDepthOutput)) / (
#                     np.max(imgDepthOutput) - np.min(imgDepthOutput))
#         colormap = cm.plasma
#         imgDepthOutput_colored = colormap(imgDepthOutput_normalized)
#         imgDepthOutput_colored = (imgDepthOutput_colored[:, :, :3] * 255).astype(np.uint8)
#
#         output_path = os.path.join(output_folder, f"processed_colored_{depth_file}")
#         cv2.imwrite(output_path, cv2.cvtColor(imgDepthOutput_colored, cv2.COLOR_RGB2BGR))
#
#         # output_path = os.path.join(output_folder, f"processed_{depth_file}")
#         # cv2.imwrite(output_path, imgDepthOutput)
#
#         print(f"处理完成: {rgb_file} 和 {depth_file}")
#
#     print("所有图像处理完成")