# -*- coding: utf-8 -*-

from __future__ import print_function

import numpy as np
import cv2
from ocrolib import morph, sl
from matplotlib import pyplot as plt 
from skimage.filters import threshold_otsu, threshold_adaptive
from ocrolib.line_cut.block import Block
from ocrolib.line_cut.geometry import filter_and_merge_overlap_boxes

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

def compute_separators_morph_vertical(binary, scale, widen=True):
    """Finds vertical black lines corresponding to column separators."""
    span = 3 #min(5,  int(scale * 0.2))

    d0 = span
    if widen:
        d1 = span + 1
    else:
        d1 = span
    thick = morph.r_dilation(binary, (d0, d1))
    vert = morph.r_opening(thick, (int(2 * scale) , 1))
    vert = morph.r_erosion(vert, (d0 // 2, span))

    return vert

def compute_separators_morph_horizontal(binary, scale, widen=True):
    """Finds vertical black lines corresponding to column separators."""
    span = 4
    d0 = span #int(max(5, scale / 5))
    if widen:
        d1 = span + 1
    else:
        d1 = span
    thick = morph.r_dilation(binary, (d1, d0))
    hor = morph.r_opening(thick, (1, int(4 * scale)))
    hor = morph.r_erosion(hor, (span, d0 // 2))

    return hor

def compute_combine_seps(binary, scale):
    hor_seps = compute_separators_morph_horizontal(binary, scale)
    ver_seps = compute_separators_morph_vertical(binary, scale)
    combine_seps = hor_seps | ver_seps

    return combine_seps

def detect_table(im_path, scale, maxsize = 10, debug_path = None):
    # folder, name = os.path.split(im_path)
    image = cv2.imread(im_path, 0)
    h, w = image.shape[:2]
    if len(image.shape) > 2 and image.shape[2] >= 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    binary = 1 - morph.thresh_sauvola(gray, k=0.05)
    junk_cc, _ = filter_junk_cc(binary, scale, maxsize)

    print('calculating combine sep...')
    combine_sep = compute_combine_seps(junk_cc, scale)
    # using closing morphology to connect disconnected edges
    close_thes = int(scale * 0.15)
    # close_thes = int(scale * 0.3)
    closed_sep = morph.r_closing(combine_sep, (close_thes, close_thes))
    # closed_sep = morph.r_dilation(combine_sep, (close_thes, close_thes))
    plt.imshow(closed_sep)
    plt.show()

    cv2.imwrite(im_path[:-4] + '_bin.png', ((1 - junk_cc) * 255).astype('uint8'))
    cv2.imwrite(im_path[:-4] + '_sep.png', (closed_sep * 255).astype('uint8'))

    labels, _ = morph.label(closed_sep)
    objects = morph.find_objects(labels)

    # result table list
    boxes = []

    for i, b in enumerate(objects):
        if sl.width(b) > maxsize * scale or sl.area(b) > scale * scale * 10 or (
                sl.aspect_normalized(b) > 6 and sl.max_dim(b) > scale * 1.5):

            density = np.sum(combine_sep[b])
            density = density / sl.area(b)

            if (sl.area(b) > scale * scale * 10 and sl.min_dim(b) > scale * 1.0 and sl.max_dim(
                    b) > scale * 8 and density < 0.4):
                # calculate projection to determine table border
                w = sl.width(b)
                h = sl.height(b)

                region = (labels[b] == i + 1).astype('uint8')

                border_pad = max(w, h)
                border_thres = scale * 2

                proj_x = np.sum(region, axis=0)
                proj_y = np.sum(region, axis=1)

                proj_x[3:] += proj_x[:-3]
                proj_y[3:] += proj_y[:-3]

                sep_x = np.sort([j[0] for j in np.argwhere(proj_x > 0.75 * h)])
                sep_y = np.sort([j[0] for j in np.argwhere(proj_y > 0.4 * w)])

                # skip if sep count < 2
                if len(sep_x) < 1 or len(sep_y) < 1: continue

                border_left, border_right, border_top, border_bottom = None, None, None, None

                if sep_x[0] < border_pad:
                    border_left = sep_x[0]
                if sep_x[-1] > w - border_pad:
                    border_right = sep_x[-1]
                if sep_y[0] < border_pad:
                    border_top = sep_y[0]
                if sep_y[-1] > h - border_pad:
                    border_bottom = sep_y[-1]

                # print_info(border_top, border_bottom, border_left, border_right)

                if all([j is not None for j in [border_top, border_bottom, border_left, border_right]]):
                    border_right = b[1].stop - b[1].start
                    boxes.append([b[1].start + border_left, b[0].start + border_top, b[1].start + border_right, b[0].start + border_bottom])
                    # boxes.append(([b[1].start, b[0].start, b[1].stop, b[0].stop]))

    return boxes


## text line segmentation using nearest neighbor
def text_line_segmentation_NN(image, scale, mask = None, use_binary = False, debug_image = None, offset = (0, 0)):

    h, w = image.shape[:2]

    if debug_image is None:
        debug_image = image
        if len(debug_image.shape) < 3:
            debug_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    if len(image.shape) > 2:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    if use_binary:
        image = morph.thresh_sauvola(image, k = 0.15) * 255

    if mask is not None:
        image = (image + mask * 255).astype('uint8')

    labels, _ = morph.label(image == 0)
    objects = morph.find_objects(labels)

    height_map = [sl.height(o) for o in objects if sl.height(o) > 6 and sl.height(o) < 100 and sl.aspect_normalized(o) < 8]
    avg_h = max(np.nan_to_num(np.mean(height_map)), scale * 0.6)

    block = Block(image, avg_h)
    words = block.getWordBoundingBoxes()

    lines = filter_and_merge_overlap_boxes(words, max(avg_h, scale * 1.2) * 0.3)
    lines = filter_and_merge_overlap_boxes(lines, max(avg_h, scale * 1.2) * 0.3, use_merge_same_line_only=True)

    offset_x, offset_y = offset

    # filter line by size
    lines = [l for l in lines if l[3] - l[1] > avg_h * 0.3 and l[3] - l[1] < avg_h * 2.5 and l[2] - l[0] > avg_h * 0.5]

    lines = [sl.pad_box(l, 0, (h, w)) for l in lines]
    lines = [[l[0] + offset_x, l[1] + offset_y, l[2] + offset_x, l[3] + offset_y] for l in lines]

    debug_image = block.paint(None)

    return lines, debug_image


if __name__ == "__main__":

    
    import os
    import glob
    import argparse
    import shutil

    parser = argparse.ArgumentParser(description="Table detect")
    parser.add_argument('--path', dest='path', type=str, help="path to images")
    args = parser.parse_args()

    print(args.path)
    save_folder = os.path.join(os.path.dirname(args.path), "debug")
    if os.path.exists(save_folder):
        shutil.rmtree(save_folder)
    os.makedirs(save_folder)

    for name in os.listdir(args.path):
        filename = os.path.join(args.path, name)
        #filename = "/tmp/New Folder 3/print2-1.jpg"
        deskewed_file = filename[:-4] + "_out.png"
        os.system('card-dewarp -mw 3000 -d "{}" -o "{}"'.format(filename, deskewed_file))
        im = cv2.imread(deskewed_file, 0)

        #lines, debug_im = text_line_segmentation_NN(im, scale=60)

        kernel = np.ones((3,3),np.uint8)
        # bin_img = ~(im > threshold_otsu(im))*1
        # block_size = 15
        # bin_img = ~threshold_adaptive(im, block_size, offset=10)*1
        # gray_img = ((bin_img)*255).astype('uint8')
        gray_img = cv2.dilate(255-im, kernel, iterations = 2)
        # gray_img = 255-cv2.morphologyEx(gray_img, cv2.MORPH_OPEN, kernel)
        # gray_img = 255-cv2.morphologyEx(255-im, cv2.MORPH_CLOSE, kernel)
        # gray_img = cv2.erode(gray_img, kernel, iterations = 1)
        # gray_img = cv2.dilate(gray_img, kernel, iterations = 1)
        gray_img = 255-cv2.erode(gray_img, kernel, iterations = 2)
        # plt.imshow(gray_img)
        # plt.show()

        gray_file = os.path.join(save_folder, name[:-4] + "_gray.png")
        cv2.imwrite(gray_file, gray_img)
        table_boxes = detect_table(gray_file, scale=25)
        print('Found {} tables'.format(len(table_boxes)))

        color_tmp = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
        for box in table_boxes:
            l, t, r, b = box
            cv2.rectangle(color_tmp, (l, t), (r, b), (0, 0, 255), 3)

        cv2.imwrite(os.path.join(save_folder, name[:-4] + "_out.png"), color_tmp)
        os.remove(deskewed_file)