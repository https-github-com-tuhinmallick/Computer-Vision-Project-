import numpy as np
import cv2
import matplotlib.pyplot as plt
import pdb


def loading_saving():
    '''
    Load Image and show it on screen
    '''
    file = '../img.png'
    img = cv2.imread(file)
    cv2.imshow('Techfak', img)
    # waits for a key event for delay of 5000 milliseconds.
    cv2.waitKey(5000)
    cv2.imwrite('../img.jpg', img)


def Resizing():
    '''
    Resize the image by a factor of 0.5 in both directions.
    '''
    file = '../img.png'
    img = cv2.imread(file)
    small = cv2.resize(img, dsize=(0, 0), fx=0.5, fy=0.5)
    cv2.imshow('Techfak', small)
    cv2.waitKey()
    cv2.imwrite('../small.jpg', img)


def color_ch():
    '''
    Create three images, one for each channel (red, green, blue)
    Note : OpenCV stores images in BGR format.
    '''
    file = '../img.png'
    img = cv2.imread(file)
    img = cv2.resize(img, dsize=(0, 0), fx=0.5, fy=0.5)
    blue_ch = np.copy(img)
    green_ch = np.copy(img)
    red_ch = np.copy(img)

    red_ch[:, :, :2] = 0
    green_ch[:, :, [0,2]] = 0
    blue_ch[:, :, 1:] = 0

    # demonstration with cv2
    # Horizontally concatenate the 3 images
    img3 = cv2.hconcat([red_ch, green_ch, blue_ch])
    cv2.imshow('Techfak', img3)
    cv2.waitKey()

    # demonstration with matplotlib
    # plt.subplot(1, 3, 1), plt.imshow(cv2.cvtColor(red_ch, cv2.COLOR_BGR2RGB))
    # plt.subplot(1, 3, 2), plt.imshow(cv2.cvtColor(green_ch, cv2.COLOR_BGR2RGB))
    # plt.subplot(1, 3, 3), plt.imshow(cv2.cvtColor(blue_ch, cv2.COLOR_BGR2RGB))
    # plt.show()





if __name__ == '__main__':
    loading_saving()
    Resizing()
    color_ch()