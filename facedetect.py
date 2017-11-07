# -*- coding: UTF-8 -*-

import sys
import os
import dlib
import glob
import numpy
import cv2

from skimage import io

size = 64
if len(sys.argv) != 5:
    print "请检查参数是否正确"
    exit()
# 1.人脸关键点检测器
predictor_path = sys.argv[1]
# 2.人脸识别模型
face_rec_model_path = sys.argv[2]
# 3.候选人脸文件夹
faces_folder_path = sys.argv[3]
# 4.需识别的人脸
img_path = sys.argv[4]

# 1.加载正脸检测器
detector = dlib.get_frontal_face_detector()
# 2.加载人脸关键点检测器
sp = dlib.shape_predictor(predictor_path)

# 3. 加载人脸识别模型
facerec = dlib.face_recognition_model_v1(face_rec_model_path)

# win = dlib.image_window()


# 候选人脸描述子list
descriptors = []
# 对文件夹下的每一个人脸进行:

# 1.人脸检测
# 2.关键点检测
# 3.描述子提取

for f in glob.glob(os.path.join(faces_folder_path, "*.jpg")):
    print("Processing file: {}".format(f))
    img = io.imread(f)
    # win.clear_overlay()
    # win.set_image(img)

    # 1.人脸检测
    dets = detector(img, 1)
    print("Number of faces detected: {}".format(len(dets)))

    for k, d in enumerate(dets):
        # 2.关键点检测
        shape = sp(img, d)
        # 画出人脸区域和和关键点
        # win.clear_overlay()
        # win.add_overlay(d)
        # win.add_overlay(shape)

        # 3.描述子提取，128D向量
        face_descriptor = facerec.compute_face_descriptor(img, shape)

        # 转换为numpy array
        v = numpy.array(face_descriptor)
        descriptors.append(v)

# 对需识别人脸进行同样处理


# 提取描述子，不再注释

# 候选人名单

candidate = ['Bingbing', 'Feifei', 'Unknown2', 'Shishi', 'douxiao', 'Unknown4', 'Unknown1']
cam = cv2.VideoCapture(0)  # 注意在这里就是打开摄像头了


while True:
    _, img = cam.read()  # 把视频转成一帧一帧的图像
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    dets = detector(gray_image, 1)
    dist = []
    if not len(dets):
        # print('Can`t get face.')
        cv2.imshow('image', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    for k, d in enumerate(dets):
        shape = sp(img, d)
        face_descriptor = facerec.compute_face_descriptor(img, shape)
        d_test = numpy.array(face_descriptor)
        x1 = d.top() if d.top() > 0 else 0
        y1 = d.bottom() if d.bottom() > 0 else 0
        x2 = d.left() if d.left() > 0 else 0
        y2 = d.right() if d.right() > 0 else 0
        face = img[x1:y1, x2:y2]
        cv2.rectangle(img, (x2, x1), (y2, y1), (255, 0, 0), 3)
        # 计算欧式距离
        for i in descriptors:
            dist_ = numpy.linalg.norm(i - d_test)
            dist.append(dist_)
        c_d = dict(zip(candidate, dist))
        cd_sorted = sorted(c_d.iteritems(), key=lambda d: d[1])
        cv2.putText(img, cd_sorted[0][0], (x2, x1), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,255), 2)
        # 各参数依次是：照片/添加的文字/左上角坐标/字体/字体大小/颜色/字体粗细
        print "\n The person is: ", cd_sorted[0][0]
        cv2.imshow('image', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
         break
#  释放摄像头
cam.release()
cv2.destroyAllWindows()




