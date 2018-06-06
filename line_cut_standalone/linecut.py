# -*- coding: utf-8 -*-

from __future__ import print_function

import numpy as np
import cv2

from block import Block
from geometry import filter_and_merge_overlap_boxes
from skimage.filters import threshold_sauvola
from scipy.ndimage.measurements import label, find_objects, sum
from util import sl
from matplotlib import pyplot as plt 

## remove junk connected components (line separators)
def filter_junk_cc(binary, scale, maxsize):
    junk_cc = np.zeros(binary.shape, dtype='B')
    text_like = np.zeros(binary.shape, dtype='B')

    labels, _ = morph.label(binary)
    objects = morph.find_objects(labels)

    for i, b in enumerate(objects):

        if sl.width(b) > maxsize * scale or sl.area(b) > scale * scale * 8 or \
                        sl.aspect_normalized(b) > 8 or sl.min_dim(b) < scale * 0.2:

            junk_cc[b][labels[b] == i + 1] = 1
        else:
            if sl.width(b) > 0.3 * scale and sl.height(b) > 0.3 * scale:
                text_like[b][labels[b] == i + 1] = 1

    return junk_cc, text_like

def remove_underline(th3, avg_h):
    # remove underline
    kernel = np.ones((1, int(3 * avg_h)), np.uint8)
    hor = cv2.erode(255 - th3, kernel, iterations=1)
    hor = cv2.dilate(hor, kernel, iterations=1)

    th3 = np.bitwise_xor(th3, hor)
    th3 = np.clip(th3, 0, 255)

    return th3

def thresh_sauvola(gray, k = 0.3, window_size = 25, auto_invert = False):
    thres = threshold_sauvola(gray, window_size = window_size, k = k)
    thres = np.nan_to_num(thres)

    bin = gray > thres

    if auto_invert and np.mean(bin) < 0.4:
        bin = np.bitwise_not(bin)

    return bin

## text line segmentation using nearest neighbor
def text_line_segmentation_NN(image, scale, use_binary = True, debug_image = None, offset = (0, 0), debug_mode = False):

    h, w = image.shape[:2]

    if debug_image is None:
        debug_image = image
        if len(debug_image.shape) < 3:
            debug_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    if len(image.shape) > 2:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    if use_binary:
        image = (thresh_sauvola(image, k = 0.2) * 255).astype('uint8')

    labels, _ = label(image == 0)
    objects = find_objects(labels)

    height_map = [sl.height(o) for o in objects if sl.height(o) > 6 and sl.height(o) < 100 and sl.aspect_normalized(o) < 8]
    avg_h = max(np.nan_to_num(np.mean(height_map)), scale * 0.6)

    # remove underline (optional)
    image = remove_underline(image, avg_h)

    block = Block(image, avg_h)
    words = block.getWordBoundingBoxes()

    words = filter_and_merge_overlap_boxes(words, max(avg_h, scale * 0.8) * 0.3, (h, w))
    words = filter_and_merge_overlap_boxes(words, max(avg_h, scale * 1.0) * 0.3, (h, w), use_merge_same_line_only=True,
                                           same_line_multiplier=3.0)

    offset_x, offset_y = offset

    # filter line by size
    lines = [(l, m) for l, m in words if l[3] - l[1] > avg_h * 0.5 and l[3] - l[1] < min(avg_h, scale * 1.5) * 3.5 and l[2] - l[0] > avg_h * 0.25
                    and 1.0 * (l[2] - l[0]) / (l[3] - l[1]) > 0.2 and max(l[3] - l[1], l[2] - l[0]) > avg_h * 0.8]
    masks = [m for _, m in lines]

    lines = [sl.pad_box(l, 0, (h, w)) for l, _ in lines]
    lines = [[l[0] + offset_x, l[1] + offset_y, l[2] + offset_x, l[3] + offset_y] for l in lines]

    if debug_mode:
        debug_image = block.paint(None)
    else:
        pre_image = debug_image.copy()
        for i, m in enumerate(masks):
            x0, y0 = lines[i][0], lines[i][1]
            x1, y1 = lines[i][2], lines[i][3]
            mh, mw = m.shape

            cv2.rectangle(pre_image, (x0, y0), (x1, y1), (0, 0, 255), 1)
            cv2.rectangle(debug_image, (x0, y0), (x1, y1), (0, 0, 255), 1)
            region = debug_image[y0: y0 + mh, x0: x0 + mw]
            region[m[:region.shape[0], :region.shape[1]] > 0] = [0, 0, 255]

        alpha = 0.75
        debug_image = cv2.addWeighted(pre_image, alpha, debug_image, 1 - alpha, 0)

    return lines, debug_image

def detect_lines(filename):

    im = cv2.imread(filename)

    lines, debug_im = text_line_segmentation_NN(im, scale=37)
    print("Found {} lines".format(len(lines)))
    # cv2.imwrite(filename[:-4] + "_out.png", debug_im)
    return lines, debug_im

    # import os
    # import glob
    # import argparse

    # parser = argparse.ArgumentParser(description="Linecut")
    # parser.add_argument('files', help="input file(s) to cut", nargs='+')
    # # parser.add_argument('--data', help="data directory", type=str, default=DEFAULT_DATA_DIR)
    # args = parser.parse_args()
    # filenames = args.files

    # for filename in filenames:

    #     path, name = os.path.split(filename)
    #     im = cv2.imread(filename)

    #     lines, debug_im = text_line_segmentation_NN(im, scale=40)
    #     print("Found {} lines".format(len(lines)))
    #     # print(lines)

    #     cv2.imwrite(filename[:-4] + "_out.png", debug_im)
