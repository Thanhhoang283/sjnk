import cv2
import numpy as np
import pytesseract
from PIL import Image, ImageSequence
from datetime import date
from unidecode import  unidecode
import imutils
import os, sys, cv2
import shutil
import glob
import matplotlib.pyplot as plt
import functools

def order_points(pts):
    # initialzie a list of coordinates that will be ordered
    # such that the first entry in the list is the top-left,
    # the second entry is the top-right, the third is the
    # bottom-right, and the fourth is the bottom-left
    rect = np.zeros((4, 2), dtype="float32")

    # the top-left point will have the smallest sum, whereas
    # the bottom-right point will have the largest sum
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    # now, compute the difference between the points, the
    # top-right point will have the smallest difference,
    # whereas the bottom-left will have the largest difference
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    # if (abs(rect[0][0] - rect[3][0]) < abs(rect[0][0] - rect[1][0])):
    #     tempx = rect[3][0]
    #     tempy = rect[3][1]
    #     for i in range(3,0,-1):
    #         rect[i][0] = rect[i-1][0]
    #         rect[i][1] = rect[i-1][1]
    #     rect[0][0] = tempx
    #     rect[0][1] = tempy
    # return the ordered coordinates

    return rect

def four_point_transform(image, pts):
    # obtain a consistent order of the points and unpack them
    # individually
    epsix = 8
    epsiy = 8
    rect = order_points(pts)
    rect[0][0] = rect[0][0] - epsix
    rect[1][0] = rect[1][0] + epsix
    rect[2][0] = rect[2][0] + epsix
    rect[3][0] = rect[3][0] - epsix

    rect[0][1] = rect[0][1] - epsiy
    rect[1][1] = rect[1][1] - epsiy
    rect[2][1] = rect[2][1] + epsiy
    rect[3][1] = rect[3][1] + epsiy
    (tl, tr, br, bl) = rect


    # compute the width of the new image, which will be the
    # maximum distance between bottom-right and bottom-left
    # x-coordiates or the top-right and top-left x-coordinates
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    # compute the height of the new image, which will be the
    # maximum distance between the top-right and bottom-right
    # y-coordinates or the top-left and bottom-left y-coordinates
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    # now that we have the dimensions of the new image, construct
    # the set of destination points to obtain a "birds eye view",
    # (i.e. top-down view) of the image, again specifying points
    # in the top-left, top-right, bottom-right, and bottom-left
    # order
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")

    # compute the perspective transform matrix and then apply it
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    # return the warped image
    return warped

def Detect_Rectangle_Text_bouding_rec(img):
    Rect = []
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    sobel = cv2.Sobel(img, cv2.CV_8U, 1, 0, ksize=1, scale=1, delta=0, borderType=cv2.BORDER_DEFAULT)
    ret, thresh = cv2.threshold(sobel, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY)
    element = cv2.getStructuringElement(cv2.MORPH_RECT, (100, 1))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, element)
    kernel = np.ones((3, 10), np.uint8)
    thresh = cv2.dilate(thresh, kernel, iterations=1)
    # cv2.imwrite("Region.jpg",thresh)
    im2, contours, hierarchy = cv2.findContours(thresh, 0, 1)

    for cnt in contours:
        contoursSize = cv2.contourArea(cnt)
        if contoursSize > 10:
            approx = cv2.approxPolyDP(cnt, epsilon=3, closed=True)

            rec = cv2.boundingRect(approx)

            Rect.append(np.array([[rec[0], rec[1]], [rec[0] + rec[2], rec[1]], [rec[0] + rec[2], rec[1]+rec[3]], [rec[0], rec[1]+rec[3]]]))

    return Rect

def Detect_Rectangle_Text(img):
    Rect = []
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    sobel = cv2.Sobel(gray, cv2.CV_8U, 1, 0, ksize=3, scale=1, delta=0, borderType=cv2.BORDER_DEFAULT)
    ret, thresh = cv2.threshold(sobel, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY)
    element = cv2.getStructuringElement(cv2.MORPH_RECT, (30,5))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, element)
    kernel = np.ones((3, 3), np.uint8)
    thresh = cv2.erode(thresh, kernel, iterations=1)
    # cv2.imwrite("Region.jpg",thresh)
    im2, contours, hierarchy = cv2.findContours(thresh,0,1)

    for cnt in contours:
        contoursSize = cv2.contourArea(cnt)
        if contoursSize>10:
            approx = cv2.approxPolyDP(cnt, epsilon=3, closed=True)

            rect = cv2.minAreaRect(approx)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            box = sorted(box, key = lambda x:x[0])

            if (box[0][1] > box[1][1]):
                tempx = box[0][0]
                tempy = box[0][1]
                box[0][0] = box[1][0]
                box[0][1] = box[1][1]
                box[1][0] = tempx
                box[1][1] = tempy
            if (box[2][1] < box[3][1]):
                tempx = box[2][0]
                tempy = box[2][1]
                box[2][0] = box[3][0]
                box[2][1] = box[3][1]
                box[3][0] = tempx
                box[3][1] = tempy
            tempx = box[1][0]
            tempy = box[1][1]
            box[1][0] = box[3][0]
            box[1][1] = box[3][1]
            box[3][0] = tempx
            box[3][1] = tempy
            # if ((box[0][0] - box[3][0]) ** 2 + (box[0][1] - box[3][1]) ** 2) * 3 < ((box[0][0] - box[1][0]) ** 2 + (box[0][1] - box[1][1]) ** 2):
            Rect.append(np.array(box))

    return Rect

def detect_circle(img, copy, color=None, thickness=None):
    try:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    except:
        pass
    img = cv2.medianBlur(img, 5)
    circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1, 20,
                               param1=40, param2=20, minRadius=50, maxRadius=100)
    if (circles is None):
        return [], copy
    circles = np.uint16(np.around(circles))

    if (color is not None):
        for i in circles[0, :]:
            # draw the outer circle
            cv2.circle(copy, (i[0], i[1]), i[2], color, thickness)
            # draw the center of the circle
            cv2.circle(copy, (i[0], i[1]), 2, color, thickness)
    return circles[0], copy

def check_rec(rec, circles):
    centerx = (rec[0][1] + rec[2][1])/2
    centery = (rec[0][0] + rec[1][0])/2
    for circle in circles:
        # print((centerx - circle[1]) ** 2 + (centery - circle[0]) ** 2,circle[2]**2 + 100)
        # print((rec[0][1] - circle[1]) ** 2 + (rec[0][0] - circle[0]) ** 2, circle[2]**2 + 100)
        if ((centerx - circle[1]) ** 2 + (centery - circle[0]) ** 2)**(1/2) < circle[2] + 20 \
                and ((rec[0][1] - circle[1]) ** 2 + (rec[0][0] - circle[0]) ** 2)**(1/2) < circle[2] + 40:
            return False
    return True

def check_rec_is_company(rec, circles):
    centerx = (rec[0][1] + rec[2][1]) / 2
    for circle in circles:
        if abs(centerx - circle[1]) < circle[2]:
            return True
    return False

def merge_rec(rec1, rec2):
    rec1[0][1] = min(rec1[0][1], rec2[0][1])
    rec1[2][1] = max(rec1[2][1], rec2[2][1])

    rec1[0][0] = min(rec1[0][0], rec2[0][0])
    rec1[1][0] = max(rec1[1][0], rec2[1][0])

    rec1[1][1] = rec1[0][1]
    rec1[3][1] = rec1[2][1]
    rec1[3][0] = rec1[0][0]
    rec1[2][0] = rec1[1][0]
    return rec1

def remove_circle(img):
    kernel = np.ones((5, 5), np.uint8)
    # circles, _ = detect_circle(img, img, color=(0, 0, 0), thickness=10)

    # temp = cv2.dilate(img, kernel, iterations=1)
    # temp = cv2.erode(temp, kernel, iterations=1)
    #
    # # img = cv2.addWeighted(img, 1, 255-temp, 1, 0)
    # img = temp/255 * img
    # img = np.uint8(img)

    # cv2.imshow("img", img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # edge_detected_image = cv2.Canny(img, 60, 200)
    #
    # # --- Find all the contours in the binary image ---
    # # edge_detected_image = cv2.medianBlur(edge_detected_image, 5)
    # _, contours, hierarchy = cv2.findContours(edge_detected_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # cnt = contours
    #
    # def check_circle_1(cont, circles):
    #     # area = cv2.contourArea(cont)
    #     rect = cv2.minAreaRect(cont)
    #     box = cv2.boxPoints(rect)
    #     box = np.int0(box)
    #     rec = sorted(box, key=lambda x: x[0])
    #     for circle in circles:
    #         if ((rec[0][1] - circle[1]) ** 2 + (rec[0][0] - circle[0]) ** 2) ** (1 / 2) < circle[2] + 50:
    #             return True
    #     return False
    #
    # def check_circle(cont, circles):
    #     # area = cv2.contourArea(cont)
    #     approx = cv2.approxPolyDP(cont, 0.01 * cv2.arcLength(cont, True), True)
    #     area = cv2.contourArea(cont)
    #     if ((len(approx) < 23) & (area > 50)):
    #         # print(approx)
    #         return True
    #     return False
    #
    # for i in cnt:
    #     # if (check_circle(i, circles) and check_circle_1(i, circles)):
    #         img = cv2.drawContours(img, i, -1, (0, 255, 0), 8)

    # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # thresh_img = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)[1]
    # thresh_img = cv2.erode(thresh_img, kernel, iterations=1)
    # ret, labels = cv2.connectedComponents(thresh_img, connectivity = 4)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thresh_img = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV)[1]
    # img = img.astype('uint8')
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(thresh_img, connectivity=8)
    height = stats[:, -2]

    satisfied_label = []
    max_size = 70
    for i in range(2, nb_components):
        if height[i] > max_size:
            satisfied_label.append(i)

    # img2 = np.zeros(output.shape)
    # for i in range(output.shape[0]):
    #     for j in range(output.shape[1]):
    #         if (output[i,j] in satisfied_label):
    #             img2[i,j] = 255

    img2 = np.isin(output, satisfied_label).astype(np.float) * 255

    img2 = cv2.dilate(img2, kernel, iterations=1)
    img2 = cv2.erode(img2, kernel, iterations=1)

    # detect_circle(img2, img2.astype(np.uint8), color=(255,0,0), thickness=)
    # for i in range(output.shape[0]):
    #     for j in range(output.shape[1]):
    #         if (img2[i,j] == 255):
    #             img[i,j,1] = 255
    #             img[i,j,2] = 255
    #             img[i,j,0] = 255

    img2 = cv2.cvtColor(img2.astype(np.uint8), cv2.COLOR_GRAY2BGR)
    # edge_detected_image = cv2.Canny(np.uint8(img2), 60, 200)
    # cv2.imshow("img", img2)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    img = cv2.addWeighted(img, 1, img2, 1, 0)
    return img

def detect_group_RecList(RecList, epsilon):
    rec_label = np.zeros(len(RecList), dtype=int)
    n_label = -1
    rec_data = []
    rec_block = []

    for i in range(len(RecList)):
        if (rec_label[i] == 0):
            n_label += 1
            rec_label[i] = n_label
            rec_data.append([RecList[i][0][1], RecList[i][2][1], RecList[i][0][0]])
            rec_block.append([RecList[i]])

        for j in range(i+1, len(RecList)):
            if (RecList[j][0][1] - RecList[i][2][1] < epsilon and not(RecList[j][0][0] > RecList[i][1][0]
                                                                      or RecList[j][1][0] < RecList[i][0][0])):
                rec_label[j] = rec_label[i]
                rec_data[rec_label[i]][1] = max(rec_data[rec_label[i]][1], RecList[j][2][1])
                rec_block[rec_label[i]].append(RecList[j])

    return n_label, rec_label, rec_data, rec_block

def remove_fluid_blob(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thresh_img = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)[1]
    kernel = np.ones((10, 10), np.uint8)
    thresh_img = cv2.erode(thresh_img, kernel, iterations=1)
    thresh_img = cv2.dilate(thresh_img, kernel, iterations=3)
    # cv2.imshow("img", thresh_img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    thresh_img = cv2.cvtColor(thresh_img.astype(np.uint8), cv2.COLOR_GRAY2BGR)
    img = cv2.addWeighted(img, 1, thresh_img, 1, 0)
    return img

def detect_lines(img_path, output_dir, epsilon, type="min"):
    img = cv2.imread(img_path)
    images = [img]

    # Output
    x = 0
    img_name = (img_path.split("/")[-1]).split(".")[0]
    folder_path = str(os.path.join(output_dir, img_name)) + "_cut/"
    print(folder_path)
    if (os.path.exists(folder_path) == False):
        os.mkdir(folder_path)
    box_lines = [str(len(images)) + '\r\n']
    last_label = 0

    for img_index, img in enumerate(images):
        newImg = img.copy()

        if (type == "min"):
            RecList = Detect_Rectangle_Text(img)
        else:
            RecList = Detect_Rectangle_Text_bouding_rec(img)

        #Sort by group
        RecList = sorted(RecList, key=lambda k: k[0][1])

        for i, rec in enumerate(RecList):
            try:
                x += 1
                print(rec)
                newImg = cv2.drawContours(newImg, [rec], 0, (0, 0, 255), 2)

                if (type=="min"):
                    cropped = four_point_transform(img, rec)
                else:
                    cropped = img[max(0,rec[0][1]-epsilon):min(img.shape[0],rec[2][1]+epsilon),max(0,rec[0][0]-4*epsilon):min(img.shape[0],rec[1][0]+4*epsilon)]

                if (cropped.shape[0] < 100):
                    cropped = cv2.resize(cropped, (int(100 * cropped.shape[1] / cropped.shape[0]), 100), cv2.INTER_LINEAR)
                cv2.imwrite(folder_path + str(x) + ".jpg", cropped)
                boxx = [max(0, rec[0][1] - epsilon), min(img.shape[0], rec[2][1] + epsilon),
                        max(0, rec[0][0] - 4 * epsilon),min(img.shape[0], rec[1][0] + 4 * epsilon)]

                line = " ".join([str(x) for x in [boxx[2], boxx[0], boxx[3], boxx[0], boxx[2], boxx[1], boxx[3], boxx[1]]]) + '\r\n'
                # line = ','.join([str(min_x),str(min_y),str(max_x),str(max_y)])+'\r\n'
                box_lines.append(line)
            except:
                pass

        cv2.imwrite(output_dir + img_name + "_" + str(img_index) + ".jpg", newImg)
        print(output_dir + img_name + " " + str(img_index) + ".jpg")
        # cv2.imwrite('rotate.jpg',imutils.rotate(img, 180))

    # Save box lines
    with open(folder_path + "box.txt", 'w') as f:
        f.writelines(box_lines)

input_dir = "/home/thanh/cinnamon/projects/Production/showadenk/word_seg/tmp/"
output_dir = "/home/thanh/cinnamon/projects/Production/showadenk/word_seg/tmp_2/"
epsilon = 2

if os.path.exists(output_dir):
    shutil.rmtree(output_dir)
os.makedirs(output_dir)

im_names = glob.glob(os.path.join(input_dir, '*.png')) + \
               glob.glob(os.path.join(input_dir, '*.jpg')) + \
               glob.glob(os.path.join(input_dir, '*.jpeg')) + \
               glob.glob(os.path.join(input_dir, '*.tif'))

for im_name in im_names:
    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    print(('Demo for {:s}'.format(im_name)))
    # if (im_name == "data/D1_Data/22.tif"):
    detect_lines(im_name, output_dir, epsilon, type='not')