from __future__ import print_function

import numpy as np
import cv2
from ocrolib import morph, sl
from ocrolib.line_cut.block import Block
from skimage.filters import threshold_otsu, threshold_adaptive
from linecut import detect_lines
# from tableseg import get_linecut_pos
# from tmp import get_linecut_pos
from ocrolib.line_cut.geometry import filter_and_merge_overlap_boxes
from matplotlib import pyplot as plt 
# from tableseg import get_linecut_pos 

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
    folder, name = os.path.split(im_path)
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
    # close_thes = int(scale * 1.5)
    closed_sep = morph.r_closing(combine_sep, (close_thes, close_thes))
    # closed_sep = morph.r_dilation(combine_sep, (close_thes, close_thes))
    # plt.imshow(closed_sep)
    # plt.show()

    if debug_path is not None:
        cv2.imwrite(os.path.join(debug_path, name[:-4] + '_bin.png'), ((1 - junk_cc) * 255).astype('uint8'))
        cv2.imwrite(os.path.join(debug_path, name[:-4] + '_sep.png'), (closed_sep * 255).astype('uint8'))

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

def get_boxes(filename, save_folder):
    print("------------------- {} -------------------".format(os.path.basename(filename)))
    path, name = os.path.split(filename)

    deskewed_file = filename[:-4] + "_out.png"
    os.system('card-dewarp -mw 3000 -d "{}" -o "{}"'.format(filename, deskewed_file))
    im = cv2.imread(deskewed_file, 0)

    kernel = np.ones((3,3),np.uint8)
    gray_img = cv2.dilate(255-im, kernel, iterations = 2)
    gray_img = 255-cv2.erode(gray_img, kernel, iterations = 2)
    # gray_img = 255-cv2.dilate(gray_img, kernel, iterations = 1)

    gray_file = os.path.join(save_folder, name[:-4] + "_gray.png")
    cv2.imwrite(gray_file, gray_img)

    table_boxes = detect_table(gray_file, scale=25, debug_path=save_folder)
    print('Found {} tables'.format(len(table_boxes)))
    print("Table positions: {}".format(table_boxes))

    mask = cv2.imread(gray_file[:-4] + "_sep.png", 0)
    # kernel = np.ones((3,3),np.uint8)
    # mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    # mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    # mask = cv2.erode(mask, kernel, iterations = 1)
    # mask = cv2.dilate(mask,kernel,iterations = 1)
    # mask = cv2.erode(mask, kernel, iterations = 2)
    # plt.imshow(mask)
    # plt.show()

    bin_img = ~(im > threshold_otsu(im))*1
    # block_size = 15
    # bin_img = ~threshold_adaptive(im, block_size, offset=10)*1
    gray_img = ((1-bin_img)*255).astype('uint8')
    cleaned_img = gray_img + mask
    cv2.imwrite(os.path.join(save_folder, name[:-4]+"_cleaned.png"), cleaned_img)

    # output_dir = os.path.dirname(filename)
    # linecuts, debug_im = detect_lines(filename[:-4]+"_cleaned.png", output_dir, epsilon=2, type='not')
    linecuts, debug_im = detect_lines(os.path.join(save_folder, name[:-4]+"_cleaned.png"))
    # linecuts, debug_im = detect_lines(filename)
    line_boxes = [[[line[0], line[1]], [line[2], line[1]], [line[2], line[3]], 
              [line[0], line[3]]] for line in linecuts]
    
    # linecuts, debug_im = detect_lines(deskewed_file)
    # cv2.imwrite(os.path.join(save_folder, name[:-4] + "_out.png"), debug_im)
    color_tmp = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
    for box in table_boxes:
        l, t, r, b = box
        cv2.rectangle(color_tmp, (l, t), (r, b), (0, 0, 255), 2)
    cv2.imwrite(os.path.join(save_folder, name[:-4]+"_rec.png"), color_tmp)
    # plt.imshow(im)
    # plt.show()

    cleaned_img = cv2.cvtColor(cleaned_img, cv2.COLOR_GRAY2BGR)
    for line in linecuts:
        # print(line)
        cleaned_img = cv2.rectangle(cleaned_img, (line[0], line[1]), (line[2], line[3]), thickness=3, color=(255,0,0))
        # cleaned_img = cv2.drawContours(cleaned_img, [i], 0, (0, 0, 255), 2)
    cv2.imwrite(os.path.join(save_folder, name[:-4]+"_debug.png"), cleaned_img)

    # # os.remove(deskewed_file[:-4]+"_sep.png")
    os.remove(filename[:-4]+"_out.png")
    # # os.remove(filename[:-4]+"_cleaned.png")
    # # os.remove(os.path.join(save_folder, name[:-4]+"_cleaned.png"))
    # # os.remove(deskewed_file[:-4]+"_bin.png")
    # # os.remove(filename[:-4]+"_debug.png")

    return table_boxes, line_boxes, im, mask, 

def get_points (points,up=0,down=1):
    result = []
    for i in range(down, len(points)):
        if ((points[i]-1) != points[i-1]):
            result.append(np.mean(points[down-1:i], dtype=int))
            down = i+1
        elif (i == (len(points)-1)):
            result.append(np.mean(points[down-1:i+1], dtype=int))
    return result


def table_analysis(table_list, image):
    results = []
    for table in table_list:
        im_table = image[table[1]-20:table[3], table[0]:table[2]]
        h, w = im_table.shape[:2]
        bin_im_table = (im_table > threshold_otsu(im_table))*1
        # print(h,w)
        # block_size = 15
        # bin_img = ~threshold_adaptive(im, block_size, offset=10)*1
        plt.imshow(bin_im_table)
        plt.show()
        projection = [sum(bin_im_table[:,i]) for i in range(w)]
        plt.plot(projection)
        plt.show()

        strokes = [point for point, value in enumerate(projection) if value > int(h/2)]
        # print(strokes)
        # print(table)
        points = get_points(strokes)
        print("#######: ", points)
        if points:
            tmp = [[[p1+table[0], table[1]], [p2+table[0], table[3]]] for p1, p2 in zip(points[:-1], points[1:])]
            # print(tmp)
            results.append([table, tmp])
    return results

def get_iou(bb1, bb2):
    # print("********")
    # print(bb1)
    # print(bb2)
    # assert bb1[0] <= bb1[2]
    # assert bb1[1] <= bb1[3]
    # assert bb2[0] <= bb2[2]
    # assert bb2[1] <= bb2[3]

    x_left = max(bb1[0], bb2[0])
    y_top = max(bb1[1], bb2[1])
    x_right = min(bb1[2], bb2[2])
    y_bottom = min(bb1[3], bb2[3])

    if x_right < x_left or y_bottom < y_top:
        # print("X 0.0")
        return 0.0

    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    bb1_area = (bb1[2] - bb1[0]) * (bb1[3] - bb1[1])
    bb2_area = (bb2[2] - bb2[0]) * (bb2[3] - bb2[1])

    # iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    iou = abs(intersection_area / float(bb2_area))

    # print(bb1_area)
    # print(bb2_area)
    # print(intersection_area)
    # print(iou)
    # assert iou >= 0.0
    # assert iou <= 1.0
    return iou

def get_line_relation(filename, save_folder):
    table_boxes, line_boxes, deskewed_im, rec_im = get_boxes(filename, save_folder)
    table_coors = table_analysis(table_boxes, rec_im)

    table_dict = {}
    for ith, coor in enumerate(table_coors):
        tmp_im = cv2.cvtColor(deskewed_im, cv2.COLOR_GRAY2BGR)
        table, columns = coor
        scores = [get_iou(table, line[0]+line[2]) for line in line_boxes]
        print("Num of columns: ", len(columns))
        for index, score in enumerate(scores):
            box = line_boxes[index][0]+line_boxes[index][2]
            # print("Box: ", box)
            # print("Column: ", columns[0][0]+columns[0][1])
            if (score > 0.9):
                table_dict[index] = [ith]
                print("*********")
                print(columns)
                print(box)
                iou_column_scores = [get_iou(col[0]+col[1], box) for col in columns]
                table_dict[index].append(iou_column_scores.index(max(iou_column_scores)))
                tmp_im = cv2.rectangle(tmp_im, (box[0], box[1]), (box[2], box[3]), thickness=2, color=(255,0,0))
                cv2.imwrite(os.path.join(line_folder, "table_{}_column_{}_\
                    line_{}.png".format(table_dict[index][0], table_dict[index][1], index)), deskewed_im[box[1]:box[3], box[0]:box[2]])
            cv2.imwrite(os.path.join(save_folder, "rec_debug.png"), tmp_im)

    for i_line, line in enumerate(line_boxes):
        box = line[0] + line[2]
        if (i_line not in table_dict.keys()):
            cv2.imwrite(os.path.join(line_folder, "nontable_line_\
                {}.png".format(i_line)), deskewed_im[box[1]:box[3], box[0]:box[2]])

    print("Num of lines in tables: ", len(table_dict))
    print("Total num of lines: ", len(line_boxes))


if __name__ == "__main__":

    import os
    import glob
    import argparse
    import shutil

    parser = argparse.ArgumentParser(description="Line cut")
    parser.add_argument('--path', dest='path', type=str, help="path to linecuts")
    args = parser.parse_args()

    print(args.path)
    save_folder = os.path.join(os.path.dirname(args.path), "debug")
    if os.path.exists(save_folder):
        shutil.rmtree(save_folder)
    os.makedirs(save_folder)

    # path = "/home/thanh/cinnamon/sjnk/linecut/tmp/*.jpg"
    for file in os.listdir(args.path):
        filename = os.path.join(args.path, file)

        line_folder = os.path.join(save_folder, "line_cut_{}".format(file[:-4]))
        if os.path.exists(line_folder):
            shutil.rmtree(line_folder)
        os.makedirs(line_folder)
        
        # get_line_relation(filename, save_folder)
        # table_boxes, line_boxes = get_boxes(filename, save_folder)
        table_boxes, line_boxes, deskewed_im, rec_im = get_boxes(filename, save_folder)
        table_coors = table_analysis(table_boxes, rec_im)

        # # tmp_im = cv2.imread(filename)
        # # tmp_im = cv2.cvtColor(deskewed_im, cv2.COLOR_GRAY2BGR)
        # table_dict = {}
        # for ith, coor in enumerate(table_coors):
        #     tmp_im = cv2.cvtColor(deskewed_im, cv2.COLOR_GRAY2BGR)
        #     table, columns = coor
        #     scores = [get_iou(table, line[0]+line[2]) for line in line_boxes]
        #     # print(scores)
        #     for index, score in enumerate(scores):
        #         box = line_boxes[index][0]+line_boxes[index][2]
        #         # print("Box: ", box)
        #         # print("Column: ", columns[0][0]+columns[0][1])
        #         if (score > 0.9):
        #             print("*********")
        #             print("Columns: ", columns)
        #             print("Box: ", box)
        #             table_dict[index] = [ith]
        #             iou_column_scores = [get_iou(col[0]+col[1], box) for col in columns]
        #             print("Column scores: ", iou_column_scores)
        #             print("Max: ", iou_column_scores.index(max(iou_column_scores)))
                    # table_dict[index].append(iou_column_scores.index(max(iou_column_scores)))
        #             # tmp_im = cv2.rectangle(tmp_im, (box[0], box[1]), (box[2], box[3]), thickness=2, color=(255,0,0))
                    
        #             cv2.imwrite(os.path.join(line_folder, "table_{}_column_{}_\
        #                 line_{}.png".format(table_dict[index][0], table_dict[index][1], index)), deskewed_im[box[1]:box[3], box[0]:box[2]])
        #     # cv2.imwrite(os.path.join(save_folder, "rec_debug.png"), tmp_im)


        # for i_line, line in enumerate(line_boxes):
        #     box = line[0] + line[2]
        #     if (i_line not in table_dict.keys()):
        #         # print(i_line)
        #         cv2.imwrite(os.path.join(line_folder, "nontable_line_{}.png".format(i_line)), deskewed_im[box[1]:box[3], box[0]:box[2]])

        
        # print("Num of lines in tables: ", len(table_dict))
        # print("Total num of lines: ", len(line_boxes))

