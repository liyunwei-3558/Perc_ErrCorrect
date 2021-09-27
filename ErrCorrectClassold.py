import numpy as np
import cv2


class Regressor:
    def __init__(self, x_lst, t_lst, lamda=0):
        '''
        x_lst:x坐标array，图片左上角为原点向下为x正方向
        t_lst:y坐标array，图片左上角为原点向右为y正方向
        lamda:正则化项
        '''
        # self.x = x
        self.h_resize = 10
        self.w_resize = 10
        self.x_lst = x_lst / self.h_resize
        self.t_lst = t_lst / self.w_resize
        self.lamda = lamda

    def calc_M(self, M, N):
        '''
        M:多项式次数
        N:拟合的点的个数
        '''
        self.M = M
        order = np.arange(M + 1)
        order = order[:, np.newaxis]
        e = np.tile(order, [1, N])
        XT = np.power(self.x_lst, e)
        X = np.transpose(XT)
        a = np.matmul(XT, X) + self.lamda * np.identity(M + 1)
        b = np.matmul(XT, self.t_lst)
        w = np.linalg.solve(a, b)
        # print("W:")
        # print(w)
        self.w = w
        return w

    def visualize(self, Image=np.zeros((590, 1920, 3), np.uint8), colors=(0, 0, 255)):
        height = Image.shape[0]
        width = Image.shape[1]
        X = np.arange(1, height / self.h_resize, step=0.01)
        x = X.copy()
        X = X[:, np.newaxis]
        order = np.arange(self.M + 1)
        e2 = np.tile(X, [1, self.M])
        X2 = np.power(X, order)
        Y = np.matmul(self.w, X2.T)
        # Plot regressed curve
        for i in np.arange(x.size):
            if Y[i] <= 0:
                continue
            cv2.circle(Image, (round(Y[i] * self.w_resize), round(x[i] * self.h_resize)), 6, (255, 255, 255))

        # Plot origin points
        for i in np.arange(self.x_lst.size):
            cv2.circle(Image, (round(self.t_lst[i] * self.w_resize), round(self.x_lst[i] * self.h_resize)), 5,
                       colors, 2)
