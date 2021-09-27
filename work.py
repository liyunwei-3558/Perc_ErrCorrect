import ErrCorrectClass as er
import numpy as np
import cv2

VisImage = np.zeros((590, 1920, 3), np.uint8)

x = np.array([20, 60, 120, 250, 370])
y = np.array([1330, 1270, 1200, 1088, 941]) + 300

A = er.Regressor(x, y, np.exp(-18))

A.calc_M(3, 5)

A.visualize(VisImage)

x = np.array([10, 20, 35, 65, 90])
y = np.array([1350, 1370, 1392, 1418, 1441]) + 455

B = er.Regressor(x, y)

B.calc_M(3, 5)

B.visualize(VisImage, (255, 0, 0))

# cv2.imshow("view_effect",VisImage)
cv2.imwrite("fig1.png", VisImage)
cv2.waitKey(0)
