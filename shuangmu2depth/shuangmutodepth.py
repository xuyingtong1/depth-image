import cv2
import numpy as np
from matplotlib import pyplot as plt

'''
https://blog.csdn.net/weixin_50174671/article/details/146220900
https://blog.csdn.net/qq_41748260/article/details/103992462
https://www.cnblogs.com/riddick/p/8486223.html

'''

# 双目相机参数类
class stereoCamera(object):
    def __init__(self):
        # 左相机内参 (K_02)
        self.cam_matrix_left = np.array([
            [9.597910e+02, 0.000000e+00, 6.960217e+02],
            [0.000000e+00, 9.569251e+02, 2.241806e+02],
            [0.000000e+00, 0.000000e+00, 1.000000e+00]
        ])
        # 右相机内参 (K_03)
        self.cam_matrix_right = np.array([
            [9.037596e+02, 0.000000e+00, 6.957519e+02],
            [0.000000e+00, 9.019653e+02, 2.242509e+02],
            [0.000000e+00, 0.000000e+00, 1.000000e+00]
        ])

        # 左右相机畸变系数:[k1, k2, p1, p2, k3]
        self.distortion_l = np.array([-3.691481e-01, 1.968681e-01, 1.353473e-03, 5.677587e-04, -6.770705e-02])
        self.distortion_r = np.array([-3.639558e-01, 1.788651e-01, 6.029694e-04, -3.922424e-04, -5.382460e-02])

        # 旋转矩阵 (R_02到R_03的旋转)
        self.R = np.array([[0.999976, 0.006985, 0.000271],
                           [-0.006985, 0.999970, -0.001607],
                           [-0.000269, 0.001608, 0.999999]])

        # 平移矩阵 (T_02到T_03的平移)
        self.T = np.array([[0.595662], [0.000290], [0.002577]])

        # 主点列坐标的差
        self.doffs = 0.0

        # 指示上述内外参是否为经过立体校正后的结果
        self.isRectified = False

    # def setMiddleBurryParams(self):
    #     self.cam_matrix_left = np.array([[3997.684, 0, 225.0],
    #                                      [0., 3997.684, 187.5],
    #                                      [0., 0., 1.]])
    #     self.cam_matrix_right = np.array([[3997.684, 0, 225.0],
    #                                       [0., 3997.684, 187.5],
    #                                       [0., 0., 1.]])
    #     self.distortion_l = np.zeros(shape=(5, 1), dtype=np.float64)
    #     self.distortion_r = np.zeros(shape=(5, 1), dtype=np.float64)
    #     self.R = np.identity(3, dtype=np.float64)
    #     self.T = np.array([[-193.001], [0.0], [0.0]])
    #     self.doffs = 131.111
    #     self.isRectified = True


# 获取畸变校正和立体校正的映射变换矩阵、重投影矩阵
def getRectifyTransform(height, width, config):
    # 读取内参和外参
    left_K = config.cam_matrix_left
    right_K = config.cam_matrix_right
    left_distortion = config.distortion_l
    right_distortion = config.distortion_r
    R = config.R
    T = config.T

    # 计算校正变换
    height = int(height)
    width = int(width)
    R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(left_K, left_distortion, right_K, right_distortion,
                                                      (width, height), R, T, alpha=-1)

    map1x, map1y = cv2.initUndistortRectifyMap(left_K, left_distortion, R1, P1, (width, height), cv2.CV_32FC1)
    map2x, map2y = cv2.initUndistortRectifyMap(right_K, right_distortion, R2, P2, (width, height), cv2.CV_32FC1)

    return map1x, map1y, map2x, map2y, Q


# 畸变校正和立体校正
def rectifyImage(image1, image2, map1x, map1y, map2x, map2y):
    rectifyed_img1 = cv2.remap(image1, map1x, map1y, cv2.INTER_LINEAR)
    rectifyed_img2 = cv2.remap(image2, map2x, map2y, cv2.INTER_LINEAR)

    return rectifyed_img1, rectifyed_img2


# 立体校正检验----画线
def draw_line(image1, image2):
    # 建立输出图像
    height = max(image1.shape[0], image2.shape[0])
    width = image1.shape[1] + image2.shape[1]

    output = np.zeros((height, width, 3), dtype=np.uint8)
    output[0:image1.shape[0], 0:image1.shape[1]] = image1
    output[0:image2.shape[0], image1.shape[1]:] = image2

    # 绘制等间距平行线
    line_interval = 50  # 直线间隔：50
    for k in range(height // line_interval):
        cv2.line(output, (0, line_interval * (k + 1)), (2 * width, line_interval * (k + 1)), (0, 255, 0), thickness=2,
                 lineType=cv2.LINE_AA)

    return output


# 视差计算
def stereoMatchSGBM(left_image, right_image, down_scale=False):
    # SGBM匹配参数设置
    if left_image.ndim == 2:
        img_channels = 1
    else:
        img_channels = 3
    blockSize = 8
    paraml = {'minDisparity': 0,
              'numDisparities': 64,
              'blockSize': blockSize,
              'P1': 8 * img_channels * blockSize ** 2,
              'P2': 32 * img_channels * blockSize ** 2,
              'disp12MaxDiff': 1,
              'preFilterCap': 63,
              'uniquenessRatio': 10,
              'speckleWindowSize': 10,
              'speckleRange': 8,
              'mode': cv2.STEREO_SGBM_MODE_SGBM_3WAY
              }

    # 构建SGBM对象
    left_matcher = cv2.StereoSGBM_create(**paraml)
    paramr = paraml
    paramr['minDisparity'] = -paraml['numDisparities']
    right_matcher = cv2.StereoSGBM_create(**paramr)

    # 计算视差图
    size = (left_image.shape[1], left_image.shape[0])
    if down_scale == False:
        disparity_left = left_matcher.compute(left_image, right_image)
        # plt.imshow(disparity_left.astype(np.float32), cmap='gray')
        # plt.show()
        disparity_right = right_matcher.compute(right_image, left_image)

    else:
        left_image_down = cv2.pyrDown(left_image)
        right_image_down = cv2.pyrDown(right_image)
        factor = left_image.shape[1] / left_image_down.shape[1]

        disparity_left_half = left_matcher.compute(left_image_down, right_image_down)
        disparity_right_half = right_matcher.compute(right_image_down, left_image_down)
        disparity_left = cv2.resize(disparity_left_half, size, interpolation=cv2.INTER_AREA)
        disparity_right = cv2.resize(disparity_right_half, size, interpolation=cv2.INTER_AREA)
        disparity_left = factor * disparity_left
        disparity_right = factor * disparity_right

    # 真实视差（因为SGBM算法得到的视差是×16的）
    trueDisp_left = disparity_left.astype(np.float32) / 16.
    trueDisp_right = disparity_right.astype(np.float32) / 16.

    return trueDisp_left, trueDisp_right


# 利用opencv函数计算深度图
def getDepthMapWithQ(disparityMap: np.ndarray, Q: np.ndarray) -> np.ndarray:
    # plt.imshow(disparityMap)
    # plt.show()
    points_3d = cv2.reprojectImageTo3D(disparityMap, Q)
    depthMap = points_3d[:, :, 2]

    reset_index = np.where(np.logical_or(depthMap < 0.0, depthMap > 65535.0))
    depthMap[reset_index] = 0

    return depthMap.astype(np.float32)


# 主函数
def main():
    # 初始化相机参数
    config = stereoCamera()
    # config.setMiddleBurryParams()  # 如果使用MiddleBurry数据集的参数，取消注释

    # 读取左右图像
    left_image = cv2.imread('0000000008left.png')
    right_image = cv2.imread('0000000008right.png')
    # # # 将图像转换为灰度图
    # left_image = cv2.cvtColor(left_image, cv2.COLOR_BGR2GRAY)
    # right_image = cv2.cvtColor(right_image, cv2.COLOR_BGR2GRAY)
    # 获取图像尺寸
    height, width = left_image.shape[:2]

    # 获取校正变换矩阵和重投影矩阵
    map1x, map1y, map2x, map2y, Q = getRectifyTransform(height, width, config)
    # plt.show(left_image)
    # plt.show()
    # cv2.imshow('Rectified Left Image', map1x)
    # cv2.imshow('Rectified Right Image', map1y)
    # cv2.imshow('Rectified Left Image', map2x)
    # cv2.imshow('Rectified Right Image', map2y)
    # cv2.imshow('Rectified Left Image', Q)
    # cv2.waitKey(0)
    # 畸变校正和立体校正
    rectifyed_left, rectifyed_right = rectifyImage(left_image, right_image, map1x, map1y, map2x, map2y)

    # # 显示校正后的图像
    # cv2.imshow('Rectified Left Image', rectifyed_left)
    # cv2.imshow('Rectified Right Image', rectifyed_right)
    # cv2.waitKey(0)
    # 绘制校正后的图像
    line_image = draw_line(left_image, right_image)
    cv2.imshow('Rectified Image', line_image)
    cv2.waitKey(0)
    cv2.imwrite('rectified_image.jpg', line_image)

    # 计算视差图
    disparity_left, disparity_right = stereoMatchSGBM(rectifyed_left, rectifyed_right)
    # Creating an object of StereoSGBM algorithm
    # minDisparity = 0
    # numDisparities = 64
    # blockSize = 8
    # disp12MaxDiff = 1
    # uniquenessRatio = 10
    # speckleWindowSize = 10
    # speckleRange = 8
    # stereo = cv2.StereoSGBM_create(minDisparity=minDisparity,
    #                                numDisparities=numDisparities,
    #                                blockSize=blockSize,
    #                                disp12MaxDiff=disp12MaxDiff,
    #                                uniquenessRatio=uniquenessRatio,
    #                                speckleWindowSize=speckleWindowSize,
    #                                speckleRange=speckleRange
    #                                )
    #
    # # Calculating disparith using the StereoSGBM algorithm
    # disp = stereo.compute(rectifyed_left, rectifyed_right).astype(np.float32)
    # disp = cv2.normalize(disp, 0, 255, cv2.NORM_MINMAX)
    plt.imshow(disparity_left, cmap='gray')
    plt.show()
    # cv2.imshow('Disparity Left', disparity_left)
    # cv2.imshow('Disparity Right', disparity_right)
    # cv2.waitKey(0)
    # # 使用Block Matching算法
    # stereo = cv2.StereoBM_create(numDisparities=128, blockSize=15)
    # disparity_left = stereo.compute(rectifyed_left, rectifyed_right)
    # disparity_right = stereo.compute(rectifyed_right, rectifyed_left)
    # # 检查视差图
    # cv2.imshow('Disparity Left', disparity_left)
    # cv2.imshow('Disparity Right', disparity_right)
    # cv2.waitKey(0)


    # # 计算深度图
    depthMap = getDepthMapWithQ(disparity_left, Q)
    #
    # # 将深度图转换为彩色图像
    # normalizedDepth = cv2.normalize(depthMap, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
    # coloredDepth = cv2.applyColorMap(normalizedDepth, cv2.COLORMAP_JET)

    # # 显示和保存彩色深度图
    cv2.imshow('Colored Depth Map', depthMap)
    cv2.waitKey(0)
    # # cv2.imwrite('colored_depth_map.png', coloredDepth)
    #
    # # 释放资源
    # cv2.destroyAllWindows()


if __name__ == '__main__':
    main()