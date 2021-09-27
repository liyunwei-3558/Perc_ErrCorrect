import ErrCorrectClass as er
import numpy as np
import cv2

VisImage = np.zeros((590, 1920, 3), np.uint8)

x = np.array([20, 60, 120, 200, 300]) + 240
# y = np.array([1330,1270,1200,1088,941])+300
y = np.array([100, 225, 333, 436, 477])
A = er.Regressor(x, y, np.exp(-18))

A.calc_M(3, 5)

A.visualize(VisImage)

x = np.array([20, 60, 120, 200, 300, 493]) + 50
y = np.array([120, 275, 493, 736, 977, 1400]) + 270

B = er.Regressor(x, y)

B.calc_M(3, 6)

B.visualize(VisImage, (255, 0, 0))

# cv2.imshow("view_effect",VisImage)
cv2.imwrite("fig2.png", VisImage)
cv2.waitKey(0)
