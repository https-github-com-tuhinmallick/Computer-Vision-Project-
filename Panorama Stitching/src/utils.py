import cv2
import pdb


def computeFeatures(image_data):
    # static thread_local Ptr<ORB> detector = cv::ORB::create(2000);
    # detector->(img, noArray(), keypoints, descriptors);
    # cout << "Found " << keypoints.size() << " ORB features on image " << id << endl;
    orb = cv2.ORB_create(nfeatures=500, scoreType=cv2.ORB_FAST_SCORE)
    keypoints, descriptors = orb.detectAndCompute(image_data['img'], None)
    print ("Found " + str(len(keypoints)) + " ORB features on image " + str(image_data['id']))

    image_data['keypoints'] = keypoints
    image_data['descriptors'] = descriptors

    return image_data

def createMatchImage(img1, img2, matches):
    img_matches = cv2.drawMatches(img1['img'], img1['keypoints'], img2['img'], img2['keypoints'], matches,
               outImg=None, matchColor=(0, 255, 0), singlePointColor=(0, 255, 0), flags=2)
    return img_matches
