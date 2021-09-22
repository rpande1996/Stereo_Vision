import numpy as np
import cv2
import matplotlib.pyplot as plt
import copy

# Cycle Groundtruth
cam0 = np.array([[5299.313, 0, 1263.818], [0, 5299.313, 977.763], [0, 0, 1]])
cam1 = np.array([[5299.313, 0, 1438.004], [0, 5299.313, 977.763], [0, 0, 1]])
baseline = 177.288
f = 5299.313

# Flowers Groundtruth
'''cam0=np.array([[4396.869,0,1353.072],[ 0, 4396.869, 989.702],[ 0, 0, 1]])
cam1=np.array([[4396.869,0,1538.86], [0, 4396.869, 989.702], [0, 0, 1]])
baseline=144.049
f=4396.869'''

# Umbrella Groundtruth
'''cam0=np.array([[5806.559, 0, 1429.219], [0, 5806.559, 993.403], [0, 0, 1]])
cam1=np.array([[5806.559, 0, 1543.51], [0, 5806.559, 993.403], [0, 0, 1]])
baseline=174.019
f=5806.559'''

# To compare both images in order to find matching points
def SiftCompare():
    Image_1 = cv2.imread('./Cycle/im0.png')
    Image_2 = cv2.imread('./Cycle/im1.png')
    #Image_1 = cv2.imread('./Flowers/im0.png')
    #Image_2 = cv2.imread('./Flowers/im1.png')
    #Image_1 = cv2.imread('./Umbrella/im0.png')
    #Image_2 = cv2.imread('./Umbrella/im1.png')
    Image_1 = cv2.resize(Image_1, (640, 480))
    Image_2 = cv2.resize(Image_2, (640, 480))
    Gr1 = cv2.cvtColor(Image_1, cv2.COLOR_BGR2GRAY)
    Gr2 = cv2.cvtColor(Image_2, cv2.COLOR_BGR2GRAY)

    Sf = cv2.xfeatures2d.SIFT_create()

    K1, D1 = Sf.detectAndCompute(Gr1, None)
    K2, D2 = Sf.detectAndCompute(Gr2, None)

    b = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    M = b.match(D1, D2)
    M = sorted(M, key=lambda x: x.distance)

    p1 = list()
    p2 = list()
    for m in M:
        p1.append(K1[m.queryIdx].pt)
        p2.append(K2[m.trainIdx].pt)
    return p1, p2, Image_1, Image_2

# To calculate the estimate fundamental matrix
def approxFunda(p1, p2):
    p1 = np.array(p1)
    p2 = np.array(p2)

    mn1 = np.sum(p1, axis=0) / p1.shape[0]
    mn2 = np.sum(p2, axis=0) / p2.shape[0]
    mn1 = np.reshape(mn1, (1, mn1.shape[0]))
    mn2 = np.reshape(mn2, (1, mn2.shape[0]))
    trnsl_p1 = p1 - mn1
    trnsl_p2 = p2 - mn2

    added_sqs1 = np.sum(trnsl_p1 ** 2, axis=1)
    added_sqs1 = np.reshape(added_sqs1, (trnsl_p1.shape[0], 1))

    mn_dist1 = np.sum(added_sqs1 ** (1 / 2), axis=0) / trnsl_p1.shape[0]
    factor1 = 2 ** (1 / 2) / mn_dist1[0]

    added_sqs2 = np.sum(trnsl_p2 ** 2, axis=1)
    added_sqs2 = np.reshape(added_sqs2, (trnsl_p2.shape[0], 1))

    mn_dist2 = np.sum(added_sqs2 ** (1 / 2), axis=0) / trnsl_p2.shape[0]
    factor2 = 2 ** (1 / 2) / mn_dist2[0]

    nmlz1 = factor1 * trnsl_p1
    nmlz2 = factor2 * trnsl_p2

    trnslmtrx1 = np.array([[1, 0, -mn1[0][0]], [0, 1, -mn1[0][1]], [0, 0, 1]])
    scmtrx1 = np.array([[factor1, 0, 0], [0, factor1, 0], [0, 0, 1]])

    trnslmtrx2 = np.array([[1, 0, -mn2[0][0]], [0, 1, -mn2[0][1]], [0, 0, 1]])
    scmtrx2 = np.array([[factor2, 0, 0], [0, factor2, 0], [0, 0, 1]])

    T1 = np.dot(scmtrx1, trnslmtrx1)
    T2 = np.dot(scmtrx2, trnslmtrx2)


    A = np.zeros((p1.shape[0], 9))
    for i in range(p1.shape[0]):
        A[i, :] = [nmlz2[i][0] * nmlz1[i][0], nmlz2[i][0] * nmlz1[i][1], nmlz2[i][0], nmlz2[i][1] * nmlz1[i][0],
                   nmlz2[i][1] * nmlz1[i][1], nmlz2[i][1], nmlz1[i][0], nmlz1[i][1], 1]

    U, S, Vt = np.linalg.svd(A)

    V = Vt.T
    V = V[:, -1]
    F = np.zeros((3, 3))
    count = 0
    for i in range(3):
        for j in range(3):
            F[i, j] = V[count]
            count += 1

    u, s, vt = np.linalg.svd(F)

    s[-1] = 0
    newS = np.zeros((3, 3))
    for i in range(3):
        newS[i, i] = s[i]

    newF = np.dot((np.dot(u, newS)), vt)

    regF = np.dot(np.dot(T2.T, newF), T1)
    regF = regF / regF[-1, -1]

    return regF


# Calculate RANSAC over the fundamental matrix
def Ransac(feature1, feature2):
    thresh = 0.05
    prsntInliers = 0
    perfF = []
    p = 0.99
    N = 2000
    count = 0
    while count < N:
        inlier_count = 0
        random_feature1 = []
        random_feature2 = []
        RandList = np.random.randint(len(feature1), size=8)
        for k in RandList:
            random_feature1.append(feature1[k])
            random_feature2.append(feature2[k])
        F = approxFunda(random_feature1, random_feature2)
        One = np.ones((len(feature1), 1))
        X_1 = np.hstack((feature1, One))
        X_2 = np.hstack((feature2, One))
        E_1, E_2 = X_1 @ F.T, X_2 @ F
        err = np.sum(E_2 * X_1, axis=1, keepdims=True) ** 2 / np.sum(np.hstack((E_1[:, :-1], E_2[:, :-1])) ** 2, axis=1,
                                                                     keepdims=True)
        Inl = err <= thresh
        InlCnt = np.sum(Inl)
        if prsntInliers < InlCnt:
            prsntInliers = InlCnt
            coor = np.where(Inl == True)
            x1_ar = np.array(feature1)
            x2_ar = np.array(feature2)
            inlier_x1 = x1_ar[coor[0][:]]
            inlier_x2 = x2_ar[coor[0][:]]

            perfF = F

        RatioInl = InlCnt / len(feature1)
        if np.log(1 - (RatioInl ** 8)) == 0: continue
        N = np.log(1 - p) / np.log(1 - (RatioInl ** 8))
        count += 1
    return perfF, inlier_x1, inlier_x2

# To calculate essential matrix
def approxF(F):
    E = np.dot(np.dot(cam1.T, F), cam0)
    U, S, Vt = np.linalg.svd(E)
    Snew = np.zeros((3, 3))
    for i in range(3):
        Snew[i, i] = 1
    Snew[-1, -1] = 0
    Enew = np.dot(np.dot(U, Snew), Vt)
    return Enew

# To calculate rotation and translation matrix
def RandT(E):
    W = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    U, S, Vt = np.linalg.svd(E)
    r_1 = np.dot(np.dot(U, W), Vt)
    c_1 = U[:, 2]
    if np.linalg.det(r_1) < 0:
        r_1 = -r_1
        c_1 = -c_1
    r_2 = np.dot(np.dot(U, W), Vt)
    c_2 = -U[:, 2]
    if np.linalg.det(r_2) < 0:
        r_2 = -r_2
        c_2 = -c_2
    r_3 = np.dot(np.dot(U, W.T), Vt)
    c_3 = U[:, 2]
    if np.linalg.det(r_3) < 0:
        r_3 = -r_3
        c_3 = -c_3
    r_4 = np.dot(np.dot(U, W.T), Vt)
    c_4 = -U[:, 2]
    if np.linalg.det(r_4) < 0:
        r_4 = -r_4
        c_4 = -c_4

    c_1 = np.reshape(c_1, (3, 1))
    c_2 = np.reshape(c_2, (3, 1))
    c_3 = np.reshape(c_3, (3, 1))
    c_4 = np.reshape(c_4, (3, 1))

    Rmat = [r_1, r_2, r_3, r_4]
    Tmat = [c_1, c_2, c_3, c_4]
    return Rmat, Tmat

# To calculate 3D points to check the Cheirality condition
def calc3Dpts(r_2, c_2, p1, p2):
    c_1 = np.array([[0], [0], [0]])
    r_1 = np.identity(3)
    r_2 = r_2.T
    r_1_c_1 = -np.dot(r_1, c_1)
    r_2_c_2 = -np.dot(r_2, c_2)

    conc_1 = np.concatenate((r_1, r_1_c_1), axis=1)
    conc_2 = np.concatenate((r_2, r_2_c_2), axis=1)

    P1 = np.dot(cam0, conc_1)
    P2 = np.dot(cam1, conc_2)

    X = list()
    for i in range(len(p1)):
        X_1 = np.array(p1[i])
        X_2 = np.array(p2[i])
        X_1 = np.reshape(X_1, (2, 1))
        b = np.array([1])
        b = np.reshape(b, (1, 1))
        X_1 = np.concatenate((X_1, b), axis=0)
        X_2 = np.reshape(X_2, (2, 1))
        X_2 = np.concatenate((X_2, b), axis=0)
        skewX1 = np.array([[0, -X_1[2][0], X_1[1][0]], [X_1[2][0], 0, -X_1[0][0]], [-X_1[1][0], X_1[0][0], 0]])
        skewX2 = np.array([[0, -X_2[2][0], X_2[1][0]], [X_2[2][0], 0, -X_2[0][0]], [-X_2[1][0], X_2[0][0], 0]])
        A1 = np.dot(skewX1, P1)
        A2 = np.dot(skewX2, P2)
        A = np.zeros((6, 4))
        for i in range(6):
            if i <= 2:
                A[i, :] = A1[i, :]
            else:
                A[i, :] = A2[i - 3, :]
        U, S, Vt = np.linalg.svd(A)
        Vt = Vt[3]
        Vt = Vt / Vt[-1]
        X.append(Vt)
    return np.array(X)

# Check the triangulation
def checkTrian(Rs, Cs, p1, p2):
    count_list = list()
    for i in range(4):
        Z = calc3Dpts(Rs[i], Cs[i], p1, p2)
        count = 0
        for j in range(Z.shape[0]):
            coor = Z[j, :].reshape(-1, 1)
            if np.dot(Rs[i][2], (coor[0:3] - Cs[i])) > 0 and coor[2] > 0:
                count += 1
        count_list.append(count)
    Indx_ = count_list.index(max(count_list))
    if Cs[Indx_][2] > 0:
        Cs[Indx_] = -Cs[Indx_]
    return Rs[Indx_], Cs[Indx_]

# To calculate Least Squares
def LeastSquares(X_11, X22_Dash):
    List = list()

    # forming the X matrix
    X = X_11
    Y = np.reshape(X22_Dash, (X22_Dash.shape[0], 1))

    # computing B matrix as (X'X)^-1 (X'Y)
    ds = np.dot(X.T, X)
    ab = np.linalg.inv(ds)
    df = np.dot(X.T, Y)
    B = np.dot(ab, df)

    # computing the y coordinates and forming a list to return
    Ynew = np.dot(X, B)
    for i in Ynew:
        for a in i:
            List.append(a)

    return B

# To rectify the image
def Rect(F, p1, p2):
    u, s, vt = np.linalg.svd(F)
    Si = np.where(s < 0.00000001)
    v = vt.T
    E_l = v[:, Si[0][0]]
    E_r = u[:, Si[0][0]]
    E_l = np.reshape(E_l, (E_l.shape[0], 1))
    E_r = np.reshape(E_r, (E_r.shape[0], 1))
    t_0 = np.array([[1, 0, -(640 / 2)], [0, 1, -(480 / 2)], [0, 0, 1]])
    e = np.dot(t_0, E_r)
    e = e[:, :] / e[-1, :]
    len = ((e[0][0]) ** (2) + (e[1][0]) ** (2)) ** (1 / 2)
    if e[0][0] >= 0:
        alpha = 1
    else:
        alpha = -1
    t_1 = np.array(
        [[(alpha * e[0][0]) / len, (alpha * e[1][0]) / len, 0], [-(alpha * e[1][0]) / len, (alpha * e[0][0]) / len, 0],
         [0, 0, 1]])
    e = np.dot(t_1, e)
    t_2 = np.array([[1, 0, 0], [0, 1, 0], [((-1) / e[0][0]), 0, 1]])
    e = np.dot(t_2, e)
    Ph2 = np.dot(np.dot(np.linalg.inv(t_0), t_2), np.dot(t_1, t_0))

    gt = np.array([1, 1, 1])
    gt = np.reshape(gt, (1, 3))
    ex = np.array([[0, -E_l[2][0], E_l[1][0]], [E_l[2][0], 0, -E_l[0][0]], [-E_l[1][0], E_l[0][0], 0]])
    M = np.dot(ex, F) + np.dot(E_l, gt)

    homo = np.dot(Ph2, M)
    b = np.ones((p1.shape[0], 1))
    p1 = np.concatenate((p1, b), axis=1)
    p2 = np.concatenate((p2, b), axis=1)
    X_11 = np.dot(homo, p1.T)
    X_11 = X_11[:, :] / X_11[2, :]
    X_11 = X_11.T
    X_22 = np.dot(Ph2, p2.T)
    X_22 = X_22[:, :] / X_22[2, :]
    X_22 = X_22.T
    X22_Dash = np.reshape(X_22[:, 0], (X_22.shape[0], 1))
    m = LeastSquares(X_11, X22_Dash)
    h_a = np.array([[m[0][0], m[1][0], m[2][0]], [0, 1, 0], [0, 0, 1]])
    Ph1 = np.dot(np.dot(h_a, Ph2), M)
    return Ph1, Ph2

# To draw epipolar line
def Line(img1, img2, lines, p1, p2):
    sh = img1.shape
    r = sh[0]
    c = sh[1]
    for r, pt1, pt2 in zip(lines, p1, p2):
        pt1 = [int(pt1[0]), int(pt1[1])]
        pt2 = [int(pt2[0]), int(pt2[1])]
        color_line = (0, 255, 0)
        color_point = (0, 0, 255)
        x0, y0 = map(int, [0, -r[2] / r[1]])
        X_1, y1 = map(int, [c, -(r[2] + r[0] * c) / r[1]])
        img1 = cv2.line(img1, (x0, y0), (X_1, y1), color_line, 1)
        img1 = cv2.circle(img1, tuple(pt1), 2, color_point, -1)
        img2 = cv2.circle(img2, tuple(pt2), 2, color_point, -1)
    return img1, img2


blsize = 7
srchblsize = 56

# To calculate sum of absolute differences
def Sad(pxval1, pxval2):
    if pxval1.shape != pxval2.shape:
        return -1

    return np.sum(abs(pxval1 - pxval2))

# To check the blocks/parts of both the images
def blockcheck(y, x, block_left, Right, blsize=5):
    # Get search range for the right image
    minX = max(0, x - srchblsize)
    maxX = min(Right.shape[1], x + srchblsize)
    first = True
    minSad = None
    minInd = None
    for x in range(minX, maxX):
        block_right = Right[y: y + blsize,
                      x: x + blsize]
        sad = Sad(block_left, block_right)
        if first:
            minSad = sad
            minInd = (y, x)
            first = False
        else:
            if sad < minSad:
                minSad = sad
                minInd = (y, x)

    return minInd

# To calculate disparity map
def dispMap(Img1, Img2):
    Left = np.asarray(Img1)
    Right = np.asarray(Img2)
    Left = Left.astype(int)
    Right = Right.astype(int)
    h, w, g = Left.shape
    dm = np.zeros((h, w))
    # Go over each pixel position
    for y in range(blsize, h - blsize):
        for x in range(blsize, w - blsize):
            block_left = Left[y:y + blsize,
                         x:x + blsize]
            minInd = blockcheck(y, x, block_left,
                                Right, blsize)
            dm[y, x] = abs(minInd[1] - x)

    return dm


P1, P2, Im1, Im2 = SiftCompare()

Fmat, X_fin_1, X_fin_2 = Ransac(P1, P2)

P1 = np.array(P1)
P2 = np.array(P2)

essential_mat = approxF(Fmat)

R0, T0 = RandT(essential_mat)
r1, t1 = checkTrian(R0, T0, P1, P2)

L1 = cv2.computeCorrespondEpilines(X_fin_2.reshape(-1, 1, 2), 2, Fmat)
L1 = L1.reshape(-1, 3)
L2 = cv2.computeCorrespondEpilines(X_fin_1.reshape(-1, 1, 2), 1, Fmat)
L2 = L2.reshape(-1, 3)

disp_img1 = copy.deepcopy(Im1)
disp_img2 = copy.deepcopy(Im2)

Im1, Im2 = Line(Im1, Im2, L1, X_fin_1, X_fin_2)
Im1, Im2 = Line(Im2, Im1, L2, X_fin_1, X_fin_2)

OneT = np.ones((X_fin_1.shape[0], 1))
X_fin_1 = np.concatenate((X_fin_1, OneT), axis=1)
X_fin_2 = np.concatenate((X_fin_2, OneT), axis=1)

Hom0, Hom1 = Rect(Fmat.T, P1, P2)

print(Hom0)
print(Hom1)

Im1_1 = cv2.warpPerspective(Im1, Hom0, (640, 480))
Im1_2 = cv2.warpPerspective(Im2, Hom1, (640, 480))

image_1 = cv2.warpPerspective(disp_img1, Hom0, (640, 480))
image_2 = cv2.warpPerspective(disp_img2, Hom1, (640, 480))

final_img1 = cv2.resize(image_1, (640, 480))
final_img2 = cv2.resize(image_2, (640, 480))

dm = dispMap(final_img1, final_img2)

cond1 = np.logical_and(dm >= 0, dm < 3)
cond2 = dm > 50

dm[cond1] = 3
dm[cond2] = 50

depth = baseline * f / dm

plt.imshow(dm, cmap='hot', interpolation='nearest')
plt.title("Disparity Map - Heat")
plt.savefig('Disparity Map - Heat.png')
plt.show()

plt.imshow(dm, cmap='gray', interpolation='nearest')
plt.title("Disparity Map - Gray")
plt.savefig('Disparity Map - Gray.png')
plt.show()

plt.imshow(depth, cmap='hot', interpolation='nearest')
plt.title("Depth  Map - Heat")
plt.savefig('Depth Map - Heat.png')
plt.show()

plt.imshow(depth, cmap='gray', interpolation='nearest')
plt.title("Depth Map - Gray")
plt.savefig('Depth Map - Gray.png')
plt.show()

both = np.hstack((Im1_1, Im1_2))

cv2.imshow("Epipolar 0", Im1_1)
cv2.imwrite("Epipolar 0.png", Im1_1)

cv2.imshow("Epipolar 1", Im1_2)
cv2.imwrite("Epipolar 1.png", Im1_2)

cv2.imshow("Combined Epipolar", both)
cv2.imwrite('Combined Epipolar.png', both)

cv2.waitKey(0)
cv2.destroyAllWindows()
