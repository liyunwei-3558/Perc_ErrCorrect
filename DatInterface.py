import numpy as np
import ErrCorrectClass as er
import ErrCorrectClassold as ero
from xml.dom.minidom import parse
import xml.dom.minidom
import cv2


class Processor:
    def __init__(self, index, filepath='./dataset/'):
        self.filepath = filepath
        self.index = index
        self.readJPEG()
        self.width = self.image.shape[1]
        self.height = self.image.shape[0]
        self.ur = []
        self.lr = []
        self.ub = []
        self.lb = []
        print("w=%d, h=%d" % (self.width, self.height))
        self.readXML()
        # self.fit()

        pass

    def readJPEG(self):
        self.image = cv2.imread(self.filepath + ("JPEGImage/" + str(self.index)) + '.jpg')
        # cv2.namedWindow("dat",cv2.WINDOW_FREERATIO)
        # cv2.imshow("dat",a)
        # cv2.waitKey(0)

    def readXML(self):
        DOMTree = xml.dom.minidom.parse(self.filepath + ("annotations/" + str(self.index)) + '.xml')

        collection = DOMTree.documentElement
        obs = collection.getElementsByTagName("object")

        for ob in obs:
            name = ob.getElementsByTagName('name')[0]
            colortag = name.childNodes[0].data

            bboxs = ob.getElementsByTagName('bndbox')[0]
            x_mid = int(bboxs.getElementsByTagName('xmin')[0].childNodes[0].data) + \
                    int(bboxs.getElementsByTagName('xmax')[0].childNodes[0].data)
            x_mid = x_mid // 2
            y_mid = int(bboxs.getElementsByTagName('ymin')[0].childNodes[0].data) + \
                    int(bboxs.getElementsByTagName('ymax')[0].childNodes[0].data)
            y_mid = y_mid // 2
            # print("x=%d, y=%d, color=%s" % (x_mid, y_mid, colortag))

            if colortag == 'r':
                if y_mid <= self.height / 2:
                    self.ur.append([x_mid, y_mid])
                elif y_mid > self.height / 2:
                    self.lr.append([x_mid, y_mid])
            elif colortag == 'b':
                if y_mid <= self.height / 2:
                    self.ub.append([x_mid, y_mid])
                elif y_mid > self.height / 2:
                    self.lb.append([x_mid, y_mid])

        # print(self.ur)

    def fit(self):
        # Upper Red
        ur = np.array(self.ur)
        x = ur[:, 0]
        y = ur[:, 1]
        A = ero.Regressor(y, x, np.exp(-10))
        A.calc_M(min(3,(len(x)+1)//2), len(x))
        # print(A.get_real_W())
        # A.visualize(self.image)
        A.visualize()

        # Upper Blue
        ub = np.array(self.ub)
        x = ub[:, 0]
        y = ub[:, 1]
        B = ero.Regressor(y, x, np.exp(-10))
        B.calc_M(min(3, (len(x) + 1) // 2), len(x))
        # print(A.get_real_W())
        B.visualize(colors=(255,50,50))
        # B.visualize(self.image,(255,50,50))
        cv2.imwrite("./fig"+str(self.index)+"_mask.png", self.image[:591,:])

    def fit_mask(self):
        mask = np.zeros_like(self.image)
        # Upper Red
        ur = np.array(self.ur)
        x = ur[:, 0]
        y = ur[:, 1]
        A = ero.Regressor(y, x, np.exp(-10))
        A.calc_M(min(3,(len(x)+1)//2), len(x))
        # print(A.get_real_W())
        # A.visualize(self.image)
        A.visualize(mask)

        # Upper Blue
        ub = np.array(self.ub)
        x = ub[:, 0]
        y = ub[:, 1]
        B = er.Regressor(y, x, np.exp(-10))
        B.calc_M(min(3, (len(x) + 1) // 2), len(x))
        # print(A.get_real_W())
        # B.visualize(colors=(255,50,50))
        B.visualize(mask,(255,50,50))
        cv2.imwrite("./fig"+str(self.index)+"_mask.png", mask[:591,:])