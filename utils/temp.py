#Reads a txt annotation, and makes everything that is not a normal temperature white
import cv2
import numpy as np
path = "/Users/QiyaoWei/Downloads/h/TestPic_2.txt"
a = np.loadtxt(path)
#print(a.shape)

#l = list()
p = "./input/bmp/TestPic_2.bmp"
img = cv2.imread(p)
for i in range(a.shape[0]):
    for j in range(a.shape[1]):
        if not(a[i][j] > 30 and a[i][j] < 37):
            img[i][j] = (255, 255, 255)
#print(len(l))
#print(l)
cv2.imshow("hi", img)
cv2.waitKey(0)
cv2.destroyAllWindows()

"""
path = "./input/TestPic_2.bmp"
import cv2
a = cv2.imread(path)
cv2.circle(a, (653, 21), 20, (255, 255, 0))
cv2.imshow("hi", a)
cv2.waitKey(0)
cv2.destroyAllWindows()
"""