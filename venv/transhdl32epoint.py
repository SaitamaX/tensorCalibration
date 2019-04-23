kittiDataPath = "E:/xbw/testbin"
pointPath = kittiDataPath + "/"
def pointProcess(pointFile):
    str = pointFile.read()
    lenStr = len(str)
    points = struct.unpack("%df"%(lenStr/4), str)
    pointsArray = np.array(points)
    pointsArray = pointsArray.reshape((-1, 4))
    return pointsArray

delta_theta_32 = 0.2
delta_fi_32 = 1.33
scale_factor = delta_fi_32 / delta_theta_32

for i in range(1000):
    pointFile = open(pointPath + "{number:06}.bin".format(number = i), "rb")
    pointsArray = pointProcess(pointFile)
    img = np.zeros((600 * 1200 * 3)).reshape(600, 1200, 3)
    for point in pointsArray:
        x = point[0]
        y = point[1]
        z = point[2]
        i = point[3]
        if x < 0:
            continue
        c = math.atan2(y, x) / (delta_theta_32 * math.pi / 180)
        r = math.atan2(z, math.sqrt(x * x + y * y)) / (delta_fi_32 * math.pi / 180)
        color = 255 * i / 20
        index_x = int(-1 * c + 600)
        index_y = int(-1 * scale_factor * r + 300)
        if index_y >= 0 and index_y < 600 and index_x >= 0 and index_x < 1200:
            img[index_y][index_x][0] = (255 - color) / 255
            img[index_y][index_x][1] = 100 / 255
            img[index_y][index_x][2] = color / 255
    cv2.imshow("IMG", img)
    cv2.waitKey(0)