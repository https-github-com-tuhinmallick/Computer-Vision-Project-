import cv2
import numpy as np
from typing import Tuple


def compute_harris_response(I: np.array, k: float = 0.06) -> Tuple[np.array]:
    """Determines the Harris Response of an Image.

    Args:
        I: A Gray-level image in float32 format.
        k: A constant changing the trace to determinant ratio.

    Returns:
        A tuple with float images containing the Harris response (R) and other intermediary images. Specifically
        (R, A, B, C, Idx, Idy).
    """
    assert I.dtype == np.float32

    # Step 1: Compute dx and dy with cv2.Sobel. (2 Lines)
    dIdx = cv2.Sobel(I, cv2.CV_32F, 1, 0)
    dIdy = cv2.Sobel(I, cv2.CV_32F, 0, 1)

    cv2.imshow("dI/dx", abs(dIdx), 1, 0)
    cv2.imshow("dI/dy", abs(dIdy), 2, 0)

    # Step 2: Ixx Iyy Ixy from Idx and Idy (3 Lines)
    Ixx = dIdx ** 2
    Iyy = dIdy ** 2
    Ixy = dIdx * dIdy


    cv2.imshow("Ixx", abs(Ixx), 0, 1)
    cv2.imshow("Iyy", abs(Iyy), 1, 1)
    cv2.imshow("Ixy", abs(Ixy), 2, 1)

    # Step 3: compute A, B, C from Ixx, Iyy, Ixy with cv2.GaussianBlur (5 Lines)
    kernelSize = (3, 3)
    sdev = 1
    A = cv2.GaussianBlur(Ixx, kernelSize, sdev)
    B = cv2.GaussianBlur(Iyy, kernelSize, sdev)
    C = cv2.GaussianBlur(Ixy, kernelSize, sdev)


    cv2.imshow("A", abs(A) * 5, 0, 1)
    cv2.imshow("B", abs(B) * 5, 1, 1)
    cv2.imshow("C", abs(C) * 5, 2, 1)

    #Step 4:  Compute the harris response with the determinant and the trace of T (see announcement) (4 lines)
    ## R = Det - k * Trace*Trace
    ## Det = A * B - C * C
    ## Trace = A + B
    k = 0.06
    trace = A + B
    det = A * B - C * C
    response = det - k * (trace ** 2)

    ## Normalize the response image
    dbg = (response - np.min(response)) / (np.max(response) - np.min(response))
    dbg = dbg.astype(np.float32)
    cv2.imshow("Harris Response", dbg, 0, 2)
    
    return R, A, B, C, Idx, Idy

    raise NotImplementedError


def detect_corners(R: np.array, threshold: float = 0.1) -> Tuple[np.array, np.array]:
    """Computes key-points from a Harris response image.

    Key points are all points where the harris response is significant and greater than its neighbors.

    Args:
        R: A float image with the harris response
        threshold: A float determining which Harris response values are significant.

    Returns:
        A tuple of two 1D integer arrays containing the x and y coordinates of key-points in the image.
    """
    """
    points = []

    maxima = peak_local_max(response, min_distance=1, threshold_abs=threshold)

    for maximum in maxima:

        points.append(cv2.KeyPoint(maximum[1], maximum[0], 1))

    return points
    """
    # Step 1 (recommended) : pad the response image to facilitate vectorization (1 line)
    R =np.pad(R, ((1, 1), (1, 1)), mode='constant', constant_values=0)

    # Step 2 (recommended) : create one image for every offset in the 3x3 neighborhood (6 lines).
    A = np.array([0, 0, 0, 1, 1, 2, 2, 2])
    B = np.array([-2, -2, -2, -1, -1, R.shape[0], R.shape[0], R.shape[0]])
    C = np.array([0, 1, 2, 0, 2, 0, 1, 2])
    D = np.array([-2, -1, R.shape[1], -2, R.shape[1], -2, -1, R.shape[1]])

    list = []

    for y_s, y_f, x_s, x_f in zip(a, b, c, d):
        list.append(R_pad[y_s:y_f, x_s:x_f])

    list = np.array(list)
    # Step 3 (recommended) : compute the greatest neighbor of every pixel (1 line)
    maxima = peak_local_max(R, min_distance=1, threshold_abs=threshold)

    # Step 4 (recommended) : Compute a boolean image with only all key-points set to True (1 line)
    B = np.where(maxima > threshold, True, False)

    # Step 5 (recommended) : Use np.nonzero to compute the locations of the key-points from the boolean image (1 line)
    point_x, point_y = tuple(map(tuple, np.nonzero(B!=0))) 
    
                                 
    return point_x, point_y
    raise NotImplementedError
    """
    def find_nearest_white(img, target):

    nonzero = np.argwhere(img == 255)

    distances = np.sqrt((nonzero[:,0] - TARGET[0]) ** 2 + (nonzero[:,1] - TARGET[1]) ** 2)

    nearest_index = np.argmin(distances)

    return nonzero[nearest_index
    """


def detect_edges(R: np.array, edge_threshold: float = -0.01, epsilon=-.01) -> np.array:
    """Computes a boolean image where edge pixels are set to True.

    Edges are significant pixels of the harris response that are a local minimum along the x or y axis.

    Args:
        R: a float image with the harris response.
        edge_threshold: A constant determining which response pixels are significant

    Returns:
        A boolean image with edge pixels set to True.
    """
    
    for x, y in zip(range(response.shape[0]), range(response.shape[1])):

        minima_x = argrelextrema(response[:, y], np.less)

        minima_y = argrelextrema(response[x], np.less)

        result[minima_x, x] =  (0, 0, 255)

        result[y, minima_y] =  (0, 0, 255)
        
        
    # Step 1 (recommended) : pad the response image to facilitate vectorization (1 line)
    R = np.pad(R, ((1, 1), (1, 1)), mode='constant', constant_values=0)

    # Step 2 (recommended) : Calculate significant response pixels (1 line)
    response = np.where(R > edge_threshold, np.inf, R)

    # Step 3 (recommended) : create two images with the smaller x-axis and y-axis neighbors respectively (2 lines).
    minima_x= np.minimum(np.minimum(R[1:-1,:-2],R[1:-1,1:-1],R[1:-1,2:])
    minima_y= np.minimum(np.minimum(R[:-2,1:-1],R[1:-1,1:-1],R[2:,1:-1])

    # Step 4 (recommended) : Calculate pixels that are lower than either their x-axis or y-axis neighbors (1 line)
    result=np.logical_or(R<=minima_y, R<=minima_x)

    # Step 5 (recommended) : Calculate valid edge pixels by combining significant and axis_minimal pixels (1 line)
    valid=no.logical_and(result,response)

    return valid
    raise NotImplementedError
    
    
