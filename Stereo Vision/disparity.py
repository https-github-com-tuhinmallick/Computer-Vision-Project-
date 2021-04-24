"""
Created on July 2020.
@author: Amin Heydarshahi <amin.heydarshahi@fau.de> https://github.com/aminheydarshahi/
"""
import numpy as np
import cv2

class Params:

    def __init__(self):
        self.segmentationThreshold = 0.7
        self.minRegionSize = 100
        self.minDisp = 0
        self.maxDisp = 14

class Disparity:

    def __init__(self):
        '''
        patchRadius: The total patch size is (2*patchRadius+1) * (2*patchRadius+1)
        initIterations: Number of random guesses in initialization
        iterations: Number of spatial propagations
        optimizeIterations: Number of random guesses in optimization step of spatial propagation
        '''
        self.patchRadius = 2
        self.initIterations = 1
        self.iterations = 1
        self.optimizeIterations = 1

        self.params = Params()
        self.showDebug = True

    def computeDisparity(self, img1, img2, minDisp, maxDisp):

        self.params.minDisp = minDisp
        self.params.maxDisp = maxDisp

        scoreF = np.zeros((img1.shape[0], img1.shape[1]))
        dispF, scoreF = self.initRandom(img1, img2, scoreF)

        for i in range(self.iterations):
            self.propagate(img1, img2, dispF, scoreF, True)
            self.randomSearch(img1, img2, dispF, scoreF)
            self.propagate(img1, img2, dispF, scoreF, False)
            self.randomSearch(img1, img2, dispF, scoreF)

        self.showDebugDisp("1_after_loop", 0, 1, dispF)

        dispF = self.thresholding(dispF, scoreF, 15)
        dispF = self.segmentation(dispF)

        return dispF


    def cost(self, img1, img2, x, y, d):

        if (y+self.patchRadius >= img1.shape[0]
                or y-self.patchRadius < 0
                or x-self.patchRadius < 0
                or x+self.patchRadius >= img1.shape[1]
                or int(x-d-self.patchRadius) < 0
                or int(x-d+self.patchRadius+1) >= img1.shape[1]):
            return 10000000

        ## TODO 1.1
        ## Compute the SAD score.
        ## - Iterate over the patch in img1
        ## - use the variable "patchRadius". Note the total patch size is (patchRadius*2+1)*(patchRadius*2+1)
        ## - sample the second image with linear interpolation
        score = 0.0
        n = 2*self.patchRadius+1
        d = int(np.round(d))
        # for xp in range(x-self.patchRadius, x+self.patchRadius+1):
            # for yp in range(y-self.patchRadius, y+self.patchRadius+1):
        patch1 = img1[(y-self.patchRadius):(y+self.patchRadius+1), (x-self.patchRadius):(x+self.patchRadius+1)]
        patch2 = img2[(y-self.patchRadius):(y+self.patchRadius+1), (x-self.patchRadius-d):(x+self.patchRadius+1-d)]
        score += np.sum(np.abs(patch1-patch2))



        return score / (n**2)

    def initRandom(self, img1, img2, scoreF):

        print("Initializing Disparity with random values in the range (" + str(self.params.minDisp) + ", " + str(
            self.params.maxDisp) + ")")
        disp = np.zeros((img1.shape[0], img1.shape[1]))
        ## TODO 1.2
        ## Random initialization of the disparity map.
        ## - For each pixel compute "initIterations" random disparity values in the range[minDisp, maxDisp]\
        ## - Choose the best disparity according to the cost function and store it in disp.
        ## - Store the best cost in scoreF


        self.showDebugDisp("0_Init", 0, 1, disp)

        return disp, scoreF

    def propagate(self, img1, img2, disp, scoreF, forward=True):

        if forward:
            print("Propagate Forward")
        else:
            print("Propagate Backward")

        ## TODO 1.3
        ## Spatial propagation
        ## - In the "forward" mode, iterate over the disparity image and sample the left and upper disparity values.
        ## - Compute the cost and choose the disparity producing the lowest cost (don't forget to compare to the current
        ## cost of that pixel)
        ## - In backward mode (forward=false) this propagation is done in opposite direction


        return disp, scoreF

    def randomSearch(self, img1, img2, disp, scoreF):

        print("Random Search")

        R0 = (self.params.maxDisp - self.params.minDisp) * 0.25

        ## TODO 1.4
        ## Local search
        ## - for each pixel sample "optimizeIterations" new disparity values at decrasing radius
        ## - Choose the best disparity according to the cost (don't forget to include the current value in the comparison)


        return disp, scoreF


    def thresholding(self, disp, scoreF, t):

        print("Thresdholding with t = " + str(t))

        ## TODO 2.1
        ## Remove all disparities with score > t


        self.showDebugDisp("2_Thresholding", 1, 1, disp)
        return disp

    def segmentation(self, disp):

        print("Filter by segmentation")

        ## TODO 2.2
        ## - Segment the image by connected regions
        ## - remove all disparities with regionSize < params.minRegionSize

        ## Implementation Hints:
        ## 1. Pick a pixel which currently does not belong to a region
        ## 2. Create new region for that pixel
        ## 3. "Grow" the region as much as possible
        ## 4. Go to 1. if there are still unassigned pixels
        ## 5. Remove disparities based on region size


        self.showDebugDisp("3_Segmentation", 1, 2, disp)

        return disp

    def consitencyCheck(self, dispL, dispR):

        print("Applying consistency check")

        ## TODO 2.3
        ## - Project disparity from left to right and check if they are consistent.


        return dispL

    def showDebugDisp(self, name, x, y, disp):
        if self.showDebug:
            vis = np.copy(disp)
            vis = vis - self.params.minDisp
            vis = vis / (self.params.maxDisp - self.params.minDisp)

            if self.params.minDisp < 0:
                vis = 1 - vis

            out = np.uint8(vis * 255.0)
            cv2.imwrite(name + ".png", out)


