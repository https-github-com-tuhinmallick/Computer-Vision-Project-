"""
Created on July 2020.
@author: Amin Heydarshahi <amin.heydarshahi@fau.de> https://github.com/aminheydarshahi/
"""
import cv2
import numpy as np
import os

from disparity import Disparity

def testScore(disparity):
    oldPatchRadius = disparity.patchRadius
    disparity.patchRadius = 1

    img1 = np.array([[15, 26, 77, 15, 72],
                     [72, 161, 37, 15, 94],
                     [15, 5, 6, 16, 72],
                     [1, 2, 3, 4, 5],
                     [73, 1, 6, 27, 71]])
    img2 = np.array([[1, 2, 3, 4, 5],
                     [7, 2, 2, 7, 2],
                     [27, 15, 15, 77, 88],
                     [76, 76, 41, 99, 0],
                     [16, 17, 211, 14, 1]])

    disps = np.array([-1.1, 0, 2.7, 1.3, 1.9])

    costs = np.zeros((5, 5))
    for i in range(5):
        for j in range(5):
            costs[i, j] = disparity.cost(img1, img2, j, i, disps[j])

    costsref = np.array([[10000000, 10000000, 10000000, 10000000, 10000000],
                         [10000000, 44.666668, 10000000, 35.344448, 10000000],
                         [10000000, 53, 10000000, 39.922226, 10000000],
                         [10000000, 55.111111, 10000000, 40.32222, 10000000],
                         [10000000, 10000000, 10000000, 10000000, 10000000]])

    err = costs - costsref
    e = np.linalg.norm(err)
    print("cost error: " + str(e))

    if (e < 1e-4):
        print("Test: SUCCESS!")
    else:
        print("Reference " + str(costsref))
        print("Your result " + str(costs))
        print("Test: FAIL")

    disparity.patchRadius = oldPatchRadius

def main():
    disparity = Disparity()
    testScore(disparity)

    print("Running disparity matcher from tsukuba dataset...")

    scale = 0.5 ## Change it to 0.2 or 0.4 for faster implementation.
    left = cv2.imread('../data/tsukuba/img1.ppm')
    left = cv2.resize(left, None, fx=scale, fy=scale)
    right = cv2.imread('../data/tsukuba/img2.ppm')
    right = cv2.resize(right, None, fx=scale, fy=scale)

    if not (len(left) > 0 or len(right) > 0):
        print("The image could not be loaded. Make sure the working directory is set to the skeleton")

    leftGray = cv2.cvtColor(left, cv2.COLOR_RGB2GRAY)
    rightGray = cv2.cvtColor(right, cv2.COLOR_RGB2GRAY)

    maxDisp = 14 * scale

    t = cv2.getTickCount()
    dispFL = disparity.computeDisparity(leftGray, rightGray, 0, maxDisp)
    disparity.showDebugDisp("4_Disp left", 2, 0, dispFL)

    disparity.showDebug = False
    dispFR = disparity.computeDisparity(rightGray, leftGray, -maxDisp, 0)
    disparity.showDebug = True
    disparity.showDebugDisp("4_Disp right", 2, 1, dispFR)
    disparity.consitencyCheck(dispFL, dispFR)
    disparity.showDebugDisp("5_Final_consistent_left", 2, 2, -dispFL)
    t2 = cv2.getTickCount()

    print("Total time for both disparity maps: " + str((t2 - t) / cv2.getTickFrequency() * 1000.0) + " ms")

if __name__ == '__main__':
    main()
