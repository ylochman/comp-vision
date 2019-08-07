import cv2
import numpy as np
from utils import drawlines, imshow, normalize2dpts, hom
from matplotlib import pyplot as plt


def run(img1, img2, N=200):
    # Detect + Describe with SIFT
    sift = cv2.xfeatures2d.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    # Match with BFMatcher (+ratio test)
    bf = cv2.BFMatcher()
    matches_ = bf.knnMatch(des1, des2, k=2)
    matches = []
    pts1 = []
    pts2 = []
    for m,n in matches_:
        if m.distance < 0.7 * n.distance:
            matches.append([m])
            pts1.append(kp1[m.queryIdx].pt)
            pts2.append(kp2[m.trainIdx].pt)
    pts1 = np.int32(pts1)
    pts2 = np.int32(pts2)
    # print(pts1.shape)
    
    N = min(N, len(matches))
    
    # Show
    draw_params = dict(flags = cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    img3 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, matches[:N], None, **draw_params)
    imshow(img3, resize_factor=0.5, title='Top {} matches'.format(N))
    
    # Find F
    X1 = np.zeros((N, 2))
    X2 = np.zeros((N, 2))
    for i in range(N):
        x1_id, x2_id = matches[i][0].queryIdx, matches[i][0].trainIdx
        X1[i] = np.array(kp1[x1_id].pt)
        X2[i] = np.array(kp2[x2_id].pt)
    X10, T1 = normalize2dpts(X1)
    X20, T2 = normalize2dpts(X2)
    
    A = np.stack((X10[:,0] * X20[:,0], X10[:,0] * X20[:,1], X10[:,0],\
              X10[:,1] * X20[:,0], X10[:,1] * X20[:,1], X10[:,1],\
              X20[:,0], X20[:,1], np.ones(N)), 1)

    # SVD
    U, D, VT = np.linalg.svd(A)
    F = VT[:,-1].reshape(3,3)
    # F should have rank 2
    U, D, VT = np.linalg.svd(F, False)
    F = U.dot(np.diag((D[0], D[1], 0)).dot(VT))

    # Denormalize
    F = T2.T.dot(F).dot(T1)
    print('Fundamental matrix:\n', F)
    
    # Show
    lines1 = cv2.computeCorrespondEpilines(pts2.reshape(-1,1,2), 2, F)
    lines1 = lines1.reshape(-1,3)
    img5, img6 = drawlines(img1, img2, lines1, pts1, pts2)
    # Find epilines corresponding to points in left image (first image) and drawing its lines on right image
    lines2 = cv2.computeCorrespondEpilines(pts1.reshape(-1,1,2), 1, F)
    lines2 = lines2.reshape(-1,3)
    img3, img4 = drawlines(img2, img1, lines2, pts2, pts1)
    imshow(img5, resize_factor=0.5, sub=(1,2,1), title='Epipolar lines in the 1st image')
    imshow(img3, resize_factor=0.5, sub=(1,2,2), title='Epipolar lines in the 2nd image')
    plt.show()

def run_opencv(img1, img2, N=20):
    sift = cv2.xfeatures2d.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img1,None)
    kp2, des2 = sift.detectAndCompute(img2,None)
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params,search_params)
    matches = flann.knnMatch(des1,des2,k=2)
    
    good = []
    pts1 = []
    pts2 = []
    # ratio test as per Lowe's paper
    for i, (m,n) in enumerate(matches):
        if m.distance < 0.8*n.distance:
            good.append(m)
            pts2.append(kp2[m.trainIdx].pt)
            pts1.append(kp1[m.queryIdx].pt)
        if len(good) == N:
            break
    pts1 = np.int32(pts1)
    pts2 = np.int32(pts2)
    F, mask = cv2.findFundamentalMat(pts1,pts2,cv2.FM_LMEDS)
    print('Fundamental matrix:\n', F)
    # We select only inlier points
    pts1 = pts1[mask.ravel()==1]
    pts2 = pts2[mask.ravel()==1]
    # print(pts1.shape)
    # Find epilines corresponding to points in right image (second image) and
    # drawing its lines on left image
    lines1 = cv2.computeCorrespondEpilines(pts2.reshape(-1,1,2), 2,F)
    lines1 = lines1.reshape(-1,3)
    img5, img6 = drawlines(img1, img2, lines1, pts1, pts2)
    # Find epilines corresponding to points in left image (first image) and
    # drawing its lines on right image
    lines2 = cv2.computeCorrespondEpilines(pts1.reshape(-1,1,2), 1,F)
    lines2 = lines2.reshape(-1,3)
    img3, img4 = drawlines(img2, img1, lines2, pts2, pts1)
    imshow(img5, resize_factor=0.5, sub=(1,2,1), title='Epipolar lines in the 1st image')
    imshow(img3, resize_factor=0.5, sub=(1,2,2), title='Epipolar lines in the 2nd image')
    plt.show()
