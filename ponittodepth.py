# 开发时间：17:0617:06
# 点云转深度图
import numpy as np
import cv2
import matplotlib.pyplot as plt


pointcloud = np.fromfile("depths/0000000005.bin",dtype=np.float32).reshape(-1,4)
pointcloud = pointcloud[:,:3].astype(np.float64)
image = cv2.imread("imgs/0000000005.png")
a = np.where(pointcloud[:,0]>=0)
pointcloud = pointcloud[a]
plt.rcParams['figure.figsize'] = (120, 35)
plt.imshow(image)
plt.show()

rotation = np.array([7.533745e-03, -9.999714e-01, -6.166020e-04, 1.480249e-02, 7.280733e-04, -9.998902e-01, 9.998621e-01, 7.523790e-03, 1.480755e-02],dtype="float").reshape(3,3)
translation = np.array([-4.069766e-03, -7.631618e-02, -2.717806e-01],dtype="float").reshape(1,3)
distortion = np.array([[0,0,0,0]],dtype="float")
camera = np.array((721.5377,0,609.5593,
        0,721.5377,172.854,
        0,0,1),dtype="float").reshape(3,3)


reTransform = cv2.projectPoints(pointcloud,rotation,translation,camera,distortion)
reTransform = reTransform[0][:,0].astype(int)
pixel = reTransform
filter = np.where((pixel[:,0]<1242)&(pixel[:,1]<375)&(pixel[:,0]>=0)&(pixel[:,1]>=0))
pixel = pixel[filter]
depth = pointcloud[:,0].reshape(-1,1)[filter]
plt.scatter(pixel[:,0], pixel[:,1] ,c=depth, s=1)
plt.imshow(image)
plt.show()


print(translation)
pointcloud = np.fromfile("depths/0000000005.bin",dtype=np.float32).reshape(-1,4)
pointcloud[:,3] = 1
transform = np.hstack((rotation,translation.reshape(3,-1)))
transform = np.vstack((transform,np.array([0,0,0,1])))
print(transform)
print(pointcloud[0])
np.dot(transform,pointcloud[0].reshape(4,-1))
for i in range(pointcloud.shape[0]):
  pointcloud[i] = np.dot(transform,pointcloud[i].reshape(4,-1)).reshape(-1,4)
print(pointcloud[0])
pointcloud=pointcloud[:,:3]
a = np.where(pointcloud[:,2]>=0)
pointcloud = pointcloud[a]
rotation=np.array([0,0,0],dtype="float").reshape(3,1)
translation=np.array([0,0,0],dtype="float").reshape(1,3)

reTransform_2=cv2.projectPoints(pointcloud,rotation,translation,camera,distortion)
reTransform_2 = reTransform_2[0][:,0].astype(int)
pixel = reTransform_2
filter = np.where((pixel[:,0]<1242)&(pixel[:,1]<375)&(pixel[:,0]>=0)&(pixel[:,1]>=0))
pixel = pixel[filter]
depth = pointcloud[:,2].reshape(-1,1)[filter]
# plt.scatter(pixel[:,0], pixel[:,1] , s=1)
# plt.imshow(image)
# plt.show()

plt.scatter(pixel[:,0], pixel[:,1] ,c=depth, s=1)
plt.imshow(image)
plt.show()
depth_image=np.full(image.shape[0:2],np.nan)
depth_image[pixel[:,1],pixel[:,0]] = depth[:,0]
# depth_image=np.full(image.shape[0:2],np.nan)
print(depth_image.shape)

normalized_depth = cv2.normalize(depth_image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

# 保存深度图
cv2.imwrite("depth_image.png", normalized_depth)