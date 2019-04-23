import cv2
import os
import math
import numpy as np
import struct
import config as cfg

#/home/jlurobot/Kitti
#   +-- object
#       +-- testing
#           +-- calib
#               +-- 000000.txt
#           +-- image_2
#               +-- 000000.png
#           +-- label_2
#               +-- 000000.txt
#           +-- velodyne
#               +-- 000000.bin
#       +-- training
#           +-- calib
#               +-- 000000.txt
#           +-- image_2
#               +-- 000000.png
#           +-- label_2
#               +-- 000000.txt
#           +-- velodyne
#               +-- 000000.bin
#           +-- planes
#               +-- 000000.txt
#       +-- train.txt
#       +-- trainval.txt
#       +-- val.txt
def calibProcess(calibFile):
    calibInfo = {}
    for line in calibFile.readlines():
        line = line.split(" ")
        if line[0] == "P2:":
            calibInfo["P2_array"] = np.array([[float(line[1]), float(line[2]), float(line[3]), float(line[4])],
                                                [float(line[5]), float(line[6]), float(line[7]), float(line[8])],
                                                [float(line[9]), float(line[10]), float(line[11]), float(line[12])]
                                            ])
            #calibInfo["P2_array"] = np.delete(calibInfo["P2_array"], 3, axis=1)
        elif line[0] == "Tr_velo_to_cam:":
            calibInfo["Tr_array"] = np.array([[float(line[1]), float(line[2]), float(line[3]), float(line[4])],
                                              [float(line[5]), float(line[6]), float(line[7]), float(line[8])],
                                              [float(line[9]), float(line[10]), float(line[11]), float(line[12])]
                                              ])
        elif line[0] == "R0_rect:":
            calibInfo["Rect0_array"] = np.array([[float(line[1]), float(line[2]), float(line[3])],
                                              [float(line[4]), float(line[5]), float(line[6])],
                                              [float(line[7]), float(line[8]), float(line[9])]
                                              ])
    return calibInfo

def imageProcess(imageFile):
    return [imageFile, ]

def labelProcess(labelFile):
    labelProcessResult = {}
    detectionLabels2d = []
    detectionLabels3d = []
    for line in labelFile.readlines():
        line = line.split(" ")
        tag_class = line[0]
        if tag_class == "DontCare":
            continue
        detectionLabel2d = []
        detectionLabel2d.append(tag_class)
        detectionLabel2d.append(int(float(line[4])))
        detectionLabel2d.append(int(float(line[5])))
        detectionLabel2d.append(int(float(line[6])))
        detectionLabel2d.append(int(float(line[7])))
        detectionLabels2d.append(detectionLabel2d)
        detectionLabel3d = []
        detectionLabel3d.append(tag_class)
        detectionLabel3d.append(float(line[8]))
        detectionLabel3d.append(float(line[9]))
        detectionLabel3d.append(float(line[10]))
        detectionLabel3d.append(float(line[11]))
        detectionLabel3d.append(float(line[12]))
        detectionLabel3d.append(float(line[13]))
        detectionLabel3d.append(float(line[14]))
        detectionLabels3d.append(detectionLabel3d)
    labelProcessResult["label2d"] = detectionLabels2d
    labelProcessResult["label3d"] = detectionLabels3d
    return labelProcessResult

def pointProcess(pointFile):
    str = pointFile.read()
    lenStr = len(str)
    points = struct.unpack("%df"%(lenStr/4), str)
    pointsArray = np.array(points)
    pointsArray = pointsArray.reshape((-1, 4))
    return pointsArray

def angle_in_limit(angle):
    # To limit the angle in -pi/2 - pi/2
    limit_degree = 5
    while angle >= np.pi / 2:
        angle -= np.pi
    while angle < -np.pi / 2:
        angle += np.pi
    if abs(angle + np.pi / 2) < limit_degree / 180 * np.pi:
        angle = np.pi / 2
    return angle

def point3dToBev(input3dPoints):
    # input3dPoints(x,y,z,i):(N,4)
    N = input3dPoints.shape[0]
    ins = input3dPoints[:, 3].reshape(N, 1)
    input3dPoints = np.concatenate((input3dPoints[:, 0:3], np.ones(N).reshape(N, 1)), axis=1)
    # (N,3) -> (N,4)
    xMax = math.fabs(input3dPoints.max(axis=0)[0])
    xMin = math.fabs(input3dPoints.min(axis=0)[0])
    if xMax < xMin:
        xMax = xMin

    yMax = math.fabs(input3dPoints.max(axis=0)[1])
    yMin = math.fabs(input3dPoints.min(axis=0)[1])
    if yMax < yMin:
        yMax = yMin

    zMax = math.fabs(input3dPoints.max(axis=0)[2])
    zMin = math.fabs(input3dPoints.min(axis=0)[2])
    if zMax < zMin:
        zMax = zMin

    trans_m = np.array([[1, 0, 0, 0],[0, 1, 0, 500],[0, 0, 1, 0],[0, 0, 0, 1]])
    scale_m = np.array([[20, 0, 0, 0], [0, -20, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    temp_trans = np.dot(trans_m, scale_m)
    R = (np.dot(temp_trans, input3dPoints.T).T)[:, 0:2]
    # R:(N,2)
    R = np.concatenate((R, input3dPoints[:, 0:3]), axis=1)
    # (N,2) -> (N,5)
    R = np.concatenate((R, ins), axis=1)
    # (N,5) -> (N,6)  R: (X,Y,x,y,z,i)
    return R

def point3dTo2D(input3dPoints, calibInfo):
    temp = []
    for point in input3dPoints:
        if point[0] >= 0:
            temp.append(point)
    N = len(temp)
    input3dPoints = np.array(temp).reshape(N, 4)

    # input N, 3
    T1 = calibInfo["Tr_array"]
    T1 = np.concatenate((T1, np.array([0, 0, 0, 1]).reshape(1, 4)), axis=0)
    T2 = np.concatenate((calibInfo["Rect0_array"], np.array([0, 0, 0]).reshape(1, 3)), axis=0)
    T2 = np.concatenate((T2, np.array([0, 0, 0, 1]).reshape(4, 1)), axis=1)
    P2 = calibInfo["P2_array"]

    N = input3dPoints.shape[0]
    print("N: %d"%N)
    ins = input3dPoints[:, 3].reshape(N, 1)
    R = np.concatenate((input3dPoints[:, 0:3], np.ones(N).reshape(N, 1)), axis=1)

    xMax = math.fabs(input3dPoints.max(axis=0)[0])
    xMin = math.fabs(input3dPoints.min(axis=0)[0])
    if xMax < xMin:
        xMax = xMin

    yMax = math.fabs(input3dPoints.max(axis=0)[1])
    yMin = math.fabs(input3dPoints.min(axis=0)[1])
    if yMax < yMin:
        yMax = yMin

    zMax = math.fabs(input3dPoints.max(axis=0)[2])
    zMin = math.fabs(input3dPoints.min(axis=0)[2])
    if zMax < zMin:
        zMax = zMin

    R = np.dot(np.dot(np.dot(P2, T2), T1), input3dPoints.T).T
    R[:, [0]] /= R[:, [2]]
    R[:, [1]] /= R[:, [2]]
    R = np.delete(R, 2, axis=1)
    R = np.concatenate((R, input3dPoints[:, 0:3]), axis=1)
    R = np.concatenate((R, ins), axis=1)

    # output (N, 6) and (1, 3)
    # (x_2d y_2d x_3d y_3d z_3d ins) (x_max y_max z_max)
    maxValues = np.array([xMax, yMax, zMax])
    resultMap = {}
    resultMap["R"] = R
    resultMap["max"] = maxValues
    return resultMap

def center3dToCorner(input3dBox):
    R1 = input3dBox[0:3].reshape(1, 3)
    R1 = R1.T
    h = input3dBox[3]
    w = input3dBox[4]
    l = input3dBox[5]
    r = angle_in_limit(-1 * input3dBox[6] - math.pi / 2)
    T3 = np.array([
        [-0.5 * l, -0.5 * l, 0.5 * l, 0.5 * l, -0.5 * l, -0.5 * l, 0.5 * l, 0.5 * l],
        [0.5 * w, -0.5 * w, -0.5 * w, 0.5 * w, 0.5 * w, -0.5 * w, -0.5 * w, 0.5 * w],
        [0, 0, 0, 0, h, h, h, h]
    ])
    T4 = np.array([
        [math.cos(r), -1 * math.sin(r), 0],
        [math.sin(r), math.cos(r), 0],
        [0, 0, 1]
    ])
    T5 = np.tile(R1, 8)
    T6 = (np.dot(T4, T3) + T5).T  # 8 corner-point's coordinates
    return T6

def boxes3dToBev(input3dBoxes):
    # input N, 7
    MeanVel2Cam = np.array(([
        [7.49916597e-03, -9.99971248e-01, -8.65110297e-04, -6.71807577e-03],
        [1.18652889e-02, 9.54520517e-04, -9.99910318e-01, -7.33152811e-02],
        [9.99882833e-01, 7.49141178e-03, 1.18719929e-02, -2.78557062e-01],
        [0, 0, 0, 1]
    ]))
    MeanRect0 = np.array(([
        [0.99992475, 0.00975976, -0.00734152, 0],
        [-0.0097913, 0.99994262, -0.00430371, 0],
        [0.00729911, 0.0043753, 0.99996319, 0],
        [0, 0, 0, 1]
    ]))

    N = input3dBoxes.shape[0]

    xyz = input3dBoxes[:, 0:3]
    xyz = np.concatenate((xyz, np.ones(N).reshape(N, 1)), axis=1)
    xyz = np.dot(np.linalg.inv(MeanVel2Cam), np.dot(np.linalg.inv(MeanRect0), xyz.T)).T
    input3dBoxes[:, 0:3] = xyz[:, 0:3]
    boxes = []
    # 1, 7
    for input3dBox in input3dBoxes:
        # 8, 3
        corners = center3dToCorner(input3dBox)
        # add one 1 col
        corners = np.concatenate((corners, np.ones(8).reshape(8, 1)), axis=1)
        # reverse y-coordinate
        boxes.append(corners)
    boxes = np.array(boxes).reshape(-1, 8, 4)
    resBoxPoints = []
    for box in boxes:
        box = np.delete(box, [4, 5, 6, 7], axis=0)
        for point in box:
            resBoxPoints.append(point)
    return point3dToBev(np.array(resBoxPoints).reshape(-1, 4))

def boxes3dTo2D(input3dBoxes, calibInfo): # 将三维框标定到二维相机坐标系中
    # input N, 7
    # 标定过程中的外参和校验矩阵取所有文件的平均值（事实上每个文件对应的标定矩阵有略微差异，但这里取平均值会使整体效果变好）
    MeanVel2Cam = np.array(([
                            [7.49916597e-03, -9.99971248e-01, -8.65110297e-04, -6.71807577e-03],
                            [1.18652889e-02, 9.54520517e-04, -9.99910318e-01, -7.33152811e-02],
                            [9.99882833e-01, 7.49141178e-03, 1.18719929e-02, -2.78557062e-01],
                            [0, 0, 0, 1]
                        ]))
    MeanRect0 = np.array(([
                            [0.99992475, 0.00975976, -0.00734152, 0],
                            [-0.0097913, 0.99994262, -0.00430371, 0],
                            [0.00729911, 0.0043753, 0.99996319, 0],
                            [0, 0, 0, 1]
                        ]))
    T1 = calibInfo["Tr_array"] # T1是外参标定矩阵
    T1 = np.concatenate((T1, np.array([0, 0, 0, 1]).reshape(1, 4)), axis=0) # 外参矩阵扩展为标准形式（该形式用于标定过程中的矩阵运算）
    T2 = np.concatenate((calibInfo["Rect0_array"], np.array([0, 0, 0]).reshape(1, 3)), axis=0) #
    T2 = np.concatenate((T2, np.array([0, 0, 0, 1]).reshape(4, 1)), axis=1)
    P2 = calibInfo["P2_array"]

    #N, 3
    N = input3dBoxes.shape[0]
    centerPoints = input3dBoxes[:, :3]
    #N, 4
    temp = np.ones((N, 1))
    centerPoints = np.concatenate((centerPoints, temp), axis=1)

    #(4, 4) * (4, N) = (4, N)
    res = np.matmul(np.linalg.inv(MeanRect0), centerPoints.T)
    #(4, 4) * (4, N) = (4, N)
    res = np.matmul(np.linalg.inv(MeanVel2Cam), res)
    #(4, N) -> (N, 4)
    res = res.T
    #(N, 4) -> (N, 3)
    res = res[:, 0:3]
    res = res.reshape(-1, 3)
    #(N, 4) h,w,l,r
    temp = input3dBoxes[:, 3:]
    #(N, 3) -> (N, 7) x,y,z,h,w,l,r in velodyne-coodinate
    res = np.concatenate((res, temp), axis=1)

    boxes2D = []
    for box in res:
        T6 = center3dToCorner(box)
        T7 = np.concatenate((T6, np.ones((8, 1))), axis=1).T
        T8 = np.dot(np.dot(T2, T1), T7)
        T9 = np.dot(P2, T8).T
        T9[:, 0] /= T9[:, 2]
        T9[:, 1] /= T9[:, 2]
        T9 = T9[:, 0:2]
        boxes2D.append(T9)
    #print(boxes2D)
    return boxes2D

def visualization(calibProcessResult, imageProcessResult, labelProcessResult, pointProcessResult):
    res = []
    detection2dImg = imageProcessResult[0]
    detection3dImg = imageProcessResult[0].copy()
    detection3dPointImg = imageProcessResult[0].copy()
    detection3dPointImg[:, :, :] = 0 # 将数据结构定义为imageProcessResult形式并将内部值初始化为0
    detection3dBevImg = np.zeros((1000, 1000, 3))
    detection3dBevImg[:, :, :] = 0

    label2d = labelProcessResult["label2d"]
    label3d = labelProcessResult["label3d"]
    for label in label2d: # 画每张图像对应的二维框和框的标注信息
        cv2.rectangle(detection2dImg, (label[1], label[2]), (label[3], label[4]), (255, 0, 255), 2)
        cv2.putText(detection2dImg, label[0], (label[1]-5, label[2]-5), cv2.FONT_HERSHEY_COMPLEX, 0.7, (255, 0, 255), 2)

    labels3d = []
    for label in label3d: # 每个三维框是一个7维向量，将7维向量顺序重排序并存储到labels3d中
        h = label[1]
        w = label[2]
        l = label[3]
        x = label[4]
        y = label[5]
        z = label[6]
        r = label[7]
        labels3d.append(np.array([x, y, z, h, w, l, r]))
    #N, 7  ->  N, 8, 2
    boxes2d = boxes3dTo2D(np.array(labels3d).reshape(-1, 7), calibProcessResult)#标签的二维框数据
    boxesBev = boxes3dToBev(np.array(labels3d).reshape(-1, 7))
    boxesBev = np.delete(boxesBev, [2, 3, 4, 5], axis=1)
    boxesBev = boxesBev.reshape(-1, 4, 2)
    for box in boxesBev:
        for i in range(4):
            cv2.line(detection3dBevImg, (int(box[i][0]), int(box[i][1])), (int(box[(i + 1) % 4][0]), int(box[(i + 1) % 4][1])), (255, 0, 255), 2)

    for box in boxes2d:
        for i in range(4):
            cv2.line(detection3dImg, (int(box[i][0]), int(box[i][1])), (int(box[(i + 1) % 4][0]), int(box[(i + 1) % 4][1])), (255, 0, 255), 2)
            cv2.line(detection3dImg, (int(box[i][0]), int(box[i][1])), (int(box[i + 4][0]), int(box[i + 4][1])), (255, 0, 255), 2)
            cv2.line(detection3dImg, (int(box[i + 4][0]), int(box[i + 4][1])), (int(box[(i + 1) % 4 + 4][0]), int(box[(i + 1) % 4 + 4][1])), (255, 0, 255), 2)

            cv2.line(detection3dPointImg, (int(box[i][0]), int(box[i][1])),(int(box[(i + 1) % 4][0]), int(box[(i + 1) % 4][1])), (255, 0, 255), 2)
            cv2.line(detection3dPointImg, (int(box[i][0]), int(box[i][1])), (int(box[i + 4][0]), int(box[i + 4][1])),(255, 0, 255), 2)
            cv2.line(detection3dPointImg, (int(box[i + 4][0]), int(box[i + 4][1])),(int(box[(i + 1) % 4 + 4][0]), int(box[(i + 1) % 4 + 4][1])), (255, 0, 255), 2)

    resultMap = point3dTo2D(pointProcessResult, calibProcessResult)
    points2d = resultMap["R"]
    maxValue = resultMap["max"]
    for point2d in points2d:
        w = detection3dPointImg.shape[1]
        h = detection3dPointImg.shape[0]
        if int(point2d[1]) >= 0 and int(point2d[1]) < h and int(point2d[0]) >=0 and int(point2d[0]) < w:
            #b,g,r
            detection3dPointImg[int(point2d[1])][int(point2d[0])][0] = 255 * (1 - point2d[5])
            detection3dPointImg[int(point2d[1])][int(point2d[0])][1] = 128
            detection3dPointImg[int(point2d[1])][int(point2d[0])][2] = 255 * 1.5 * point2d[5]

    points2d = point3dToBev(pointProcessResult)
    # points2d(X,Y,x,y,z,i):(N,6)
    for point2d in points2d:
        w = detection3dBevImg.shape[1]
        h = detection3dBevImg.shape[0]
        if int(point2d[1]) >= 0 and int(point2d[1]) < h and int(point2d[0]) >= 0 and int(point2d[0]) < w: # 若该点在图片内，则在该位置上色
            # b,g,r
            detection3dBevImg[int(point2d[1])][int(point2d[0])][0] = 1.0 * (1 - point2d[5])
            detection3dBevImg[int(point2d[1])][int(point2d[0])][1] = 0.5
            detection3dBevImg[int(point2d[1])][int(point2d[0])][2] = 0.2 * 1.5 * point2d[5]
    res.append(detection2dImg)
    res.append(detection3dImg)
    res.append(detection3dPointImg)
    res.append(detection3dBevImg)
    return res

kittiDataPath = "E:/xbw/kitti_data"
# calibPath的文件存放各种标定需要的矩阵（内参、外参、校正等等）
# imagePath的文件存放各种图片数据
# labelPath的文件存放每张图片对应的标签框的数据
# pointPath的文件存放每张图片对应的三维激光雷达点云数据
calibPath = kittiDataPath + "/calib/training/calib/"
imagePath = kittiDataPath + "/image/training/image_2/"
labelPath = kittiDataPath + "/label/training/label_2/"
pointPath = kittiDataPath + "/point/training/velodyne/"

calibLenth = len([name for name in os.listdir(calibPath) if os.path.isfile(os.path.join(calibPath, name))])
imageLenth = len([name for name in os.listdir(imagePath) if os.path.isfile(os.path.join(imagePath, name))])
labelLenth = len([name for name in os.listdir(labelPath) if os.path.isfile(os.path.join(labelPath, name))])
pointLenth = len([name for name in os.listdir(pointPath) if os.path.isfile(os.path.join(pointPath, name))])

if not (calibLenth == imageLenth == labelLenth == pointLenth):
    print("Broken kitti data: The amounts of files in multi-folders are not unified.")
    print("Calib files count: {0}".format(calibLenth))
    print("Image files count: {0}".format(imageLenth))
    print("Label files count: {0}".format(labelLenth))
    print("Point files count: {0}".format(pointLenth))
    os._exit(1)
else:
    print("The amount of files is {0}".format(calibLenth))
# 读取文件前检查文件是否符合要求
for index in range(calibLenth):
    calibFile = open(calibPath + "{number:06}.txt".format(number=index), "r")
    imageFile = cv2.imread(imagePath + "{number:06}.png".format(number=index))
    labelFile = open(labelPath + "{number:06}.txt".format(number=index), "r")
    pointFile = open(pointPath + "{number:06}.bin".format(number=index), "rb")

    calibProcessResult = calibProcess(calibFile)
    imageProcessResult = imageProcess(imageFile)
    labelProcessResult = labelProcess(labelFile)
    pointProcessResult = pointProcess(pointFile)
    calibFile.close()
    labelFile.close()
    pointFile.close()
# 将文件内的各种数据存放到相应的数据结构中
    visualizationResult = visualization(calibProcessResult, imageProcessResult, labelProcessResult, pointProcessResult)

    cv2.imshow("2D detection in img", visualizationResult[0])
    cv2.imshow("3D detection in img", visualizationResult[1])
    cv2.imshow("3D detection in points fv", visualizationResult[2])
    cv2.imshow("3D detection in points bev", visualizationResult[3])


    calibFile.close()
    labelFile.close()
    pointFile.close()

    cv2.waitKey(0)