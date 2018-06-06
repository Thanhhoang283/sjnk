import cv2
import numpy
import math
import matplotlib.pyplot as plt

import colors
import geometry as g
from box import Box
from dimension import Dimension
from scipy import spatial
from scipy.ndimage.morphology import binary_closing

def threshold(image, threshold=colors.greyscale.MID_GREY, method=cv2.THRESH_BINARY_INV):
    retval, dst = cv2.threshold(image, threshold, colors.greyscale.WHITE, method)
    return dst

class Character:

    def __init__(self, x, y, box):

        self.coordinate = [x, y]
        self.x = x
        self.y = y
        self.box = g.Rect((box[0], box[1]), (box[0] + box[2], box[1] + box[3]))

        self.nearestNeighbours = []
        self.parentWord = None

    def assignParentWord(self, word):

        self.parentWord = word
        self.parentWord.registerChildCharacter(self)

        for neighbour in self.nearestNeighbours:
            if neighbour.parentWord == None:
                neighbour.assignParentWord(self.parentWord)

    def toArray(self):
        return self.coordinate

    def __len__(self):
        return len(self.coordinate)

    def __getitem__(self, key):
        return self.coordinate.__getitem__(key)

    def __setitem__(self, key, value):
        self.coordinate.__setitem__(key, value)

    def __delitem__(self, key):
        self.coordinate.__delitem__(key)

    def __iter__(self):
        return self.coordinate.__iter__()

    def __contains__(self, item):
        return self.coordinate.__contains__(item)

    def paint(self, image, color=colors.YELLOW, paint_by_point = True):

        if paint_by_point:
            pointObj = g.Point(self.coordinate)
            image = pointObj.paint(image, color)
            # cv2.rectangle(image, self.box.top_left, self.box.bottom_right, color, thickness=1)
        else:
            cv2.rectangle(image, self.box.top_left, self.box.bottom_right, color, thickness=cv2.FILLED)

        return image

class CharacterSet:

    def __init__(self, sourceImage, scale):

        self.scale = scale
        self.characters = self.getCharacters(sourceImage)
        if len(self.characters) > 0:
            self.NNTree = spatial.KDTree([char.toArray() for char in self.characters])

    def getCharacters(self, sourceImage):

        characters = []

        image = sourceImage.copy()
        image = threshold(image)
        image = threshold(image, cv2.THRESH_OTSU, method=cv2.THRESH_BINARY)

        if 1.0 * image.shape[0] / self.scale > 4:
            kernel = numpy.ones((int(max(self.scale * 0.08, 2)), 1), numpy.uint8)
            image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
            # print('closing thres', int(max(self.scale * 0.08, 2)))

        self.image = image

        contours = self.getContours(image)

        contourAreas = []
        contourSizes = []

        for contour in contours:
            box = Box(contour)
            if (min(box.width, box.height) > 0):
                aspectRatio = max(box.width, box.height) / min(box.width, box.height)
            else:
                aspectRatio = 10

            if (box.area > 50 and aspectRatio < 5):
                contourAreas.append(box.area)
                if self.scale > 0:
                    contourSizes.append(min(max(box.width, box.height), 1.8 * self.scale))
                else:
                    contourSizes.append(max(box.width, box.height))

        # test plot for freq distribution of contours area
        '''plt.plot(contourAreas)
        plt.ylabel("test")
        plt.show()'''

        medianArea = max(numpy.median(contourAreas), (self.scale ** 2) * 0.5)
        medianCharSize = max(numpy.median(contourSizes), self.scale * 0.8)

        self.medianArea = medianArea
        self.medianCharSize = medianCharSize

        # print("Median char area {}".format(medianArea))
        # print("Median char size {}".format(medianCharSize))

        for contour in contours:
            try:
                box = Box(contour)

                rect = cv2.boundingRect(contour)
                if len(contour) > 2:
                    moments = cv2.moments(contour)
                    centroidX = int( moments['m10'] / moments['m00'] )
                    centroidY = int( moments['m01'] / moments['m00'] )
                    character = Character(centroidX, centroidY, rect)
                elif len(contour) > 1:
                    centroidX = (contour[0][0][0] + contour[1][0][0]) // 2
                    centroidY = (contour[0][0][1] + contour[1][0][1]) // 2
                    character = Character(centroidX, centroidY, rect)
                else:
                    centroidX = contour[0][0][0]
                    centroidY = contour[0][0][1]
                    character = Character(centroidX, centroidY, rect)

            except ZeroDivisionError:
                continue

            thresMax = 10

            if max(box.width, box.height) > medianCharSize * 0.4 and box.area < medianArea * thresMax and max(box.width, box.height) < medianCharSize * thresMax:
                # split double char from 2 line:
                # if 1.0 * abs(character.box.width() - medianCharSize) / medianCharSize < 0.25 and 1.0 * abs(character.box.height() - 2 * medianCharSize) / medianCharSize < 0.4:
                #     x1, y1, x2, y2 = character.box
                #     hw, hh = (x2 - x1) // 2, (y2 - y1) // 2
                #     characters.append(Character(x1 + hw, y1 + hh // 2, (x1,y1,2 * hw,hh)))
                #     characters.append(Character(x1 + hw, y1 + 3 * hh // 2, (x1, y1 + hh, 2 * hw, hh)))
                #
                # else:
                characters.append(character)
            # else:
            #     if (max(box.width, box.height) > medianCharSize * 1.5 or min(box.width, box.height) < medianCharSize / 10) and abs(box.angle) < 5:
            #         junk_contours.append(contour)

        # image = cv2.bitwise_not(image)
        # image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        # junk_contours = []

        #cv2.drawContours(image, junk_contours, -1, (0,200,0), 4)
        #cv2.imwrite("junk.png", image)

        return self.splitDoubleChar(characters)

    def getContours(self, sourceImage, threshold=-1):

        image = sourceImage.copy()
        blobs = []
        topLevelContours = []

        try:

            _, contours, hierarchy = cv2.findContours(
                image, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

            for i in range(len(hierarchy[0])):

                if len(contours[i]) > 0:
                    # 1- and 2-point contours have a divide-by-zero error
                    # in calculating the center of mass.

                    # bind each contour with its corresponding hierarchy
                    # context description.
                    obj = {'contour': contours[i], 'context': hierarchy[0][i]}
                    blobs.append(obj)

            for blob in blobs:
                parent = blob['context'][3]
                if parent <= threshold: # no parent, therefore a root
                    topLevelContours.append(blob['contour'])

        except TypeError as e:
            pass

        return topLevelContours

    def splitDoubleChar(self, characters):
        if len(characters) < 2:
            return characters

        NNTree = spatial.KDTree([char.toArray() for char in characters])
        new_characters = []

        # split merged double char
        for character in characters:
            queryResult = NNTree.query(character.coordinate, k=5)
            distances = queryResult[0]
            neighbours = queryResult[1]
            character_aspect = 1.0 * character.box.height() / character.box.width()

            if abs(character_aspect - 2) < 0.5 and character.box.width() > self.medianCharSize * 0.5:
                x1, y1, x2, y2 = character.box

                # check if nearest neighbor is half the size
                nearest_half_height_left, nearest_half_height_right = False, False
                for i in range(1, len(neighbours)):
                    if distances[i] < self.medianCharSize * 2.5:
                        neighbour = characters[neighbours[i]]
                        y_diff = abs(neighbour.y - character.y)
                        if y_diff <  self.medianCharSize * 1.3 and \
                                    abs(1.0 * neighbour.box.height() / character.box.width() - 1) < 0.4:
                            if neighbour.x - character.x > 0:
                                nearest_half_height_right = True
                            else:
                                nearest_half_height_left = True

                if nearest_half_height_left and nearest_half_height_right:
                    hw, hh = (x2 - x1) // 2, (y2 - y1) // 2
                    new_characters.append(Character(x1 + hw, y1 + hh // 2, (x1, y1, 2 * hw, hh)))
                    new_characters.append(Character(x1 + hw, y1 + 3 * hh // 2, (x1, y1 + hh, 2 * hw, hh)))
                else:
                    new_characters.append(character)
            else:
                new_characters.append(character)

        return new_characters

    def getWords(self):

        if len(self.characters) == 0:
            return []

        words = []

        # find the average distance between nearest neighbours
        NNDistances = []
        for character in self.characters:
            result = self.NNTree.query(character.toArray(), k=2)  # we only want nearest neighbour, but the first result will be the point matching itself.
            nearestNeighbourDistance = result[0][1]
            NNDistances.append(nearestNeighbourDistance)

                        # min(sum(NNDistances)/len(NNDistances), self.medianCharSize)
        avgNNDistance = numpy.median(NNDistances)
        maxDistance = avgNNDistance * 2

        if avgNNDistance > self.medianCharSize * 0.5:
            avgNNDistance = min(numpy.median(NNDistances), self.medianCharSize * 0.8)
            dense_text = False
        else:
            dense_text = True

        maxDistance = min(maxDistance, self.medianCharSize * 2.5)

        # print('avg dist', avgNNDistance)

        k = 6
        for character in self.characters:
            queryResult = self.NNTree.query(character.coordinate, k=k)
            distances = queryResult[0]
            neighbours = queryResult[1]

            char_w = character.box[2] - character.box[0]
            char_h = character.box[3] - character.box[1]
            char_aspect = 1.0 * char_w / char_h


            for i in range(1,k):
                if distances[i] < maxDistance or (distances[i] < char_w * 1.6 and not dense_text):
                    neighbour = self.characters[neighbours[i]]

                    y_diff = abs(neighbour.y - character.y)
                    x_diff = abs(neighbour.x - character.x)

                    neighbour_h = neighbour.box[3] - neighbour.box[1]
                    neighbour_w = character.box[2] - character.box[0]

                    size_diff_w = abs(character.box[2] - character.box[0] - (neighbour.box[3] - neighbour.box[1]))
                    size_diff_h = abs(character.box[3] - character.box[1] - (neighbour.box[3] - neighbour.box[1]))
                    neighbour_aspect = 1.0 * (neighbour.box[2] - neighbour.box[0]) / (neighbour.box[3] - neighbour.box[1])

                    char_based_thres = min(char_h, neighbour_h) * 0.3

                    if (y_diff < avgNNDistance * 0.6
                            and ((size_diff_h < max(self.medianCharSize * 0.3, char_based_thres)) or (char_h < self.medianCharSize * 0.6 and neighbour_h < self.medianCharSize * 1.3)
                                 or (char_h < neighbour_h * 0.3 and neighbour_h < self.medianCharSize * 2 and char_aspect > 2 and character.box.intersectY(neighbour.box) > char_h)
                                 or (size_diff_w < self.medianCharSize * 0.3 and max(char_h, neighbour_h) < self.medianCharSize * 1.6))
                            and (min(char_h, neighbour_h) > self.medianCharSize * 0.4
                                    or ((x_diff > self.medianCharSize * 0.5 and character.box.intersectX(neighbour.box) < 2 and y_diff < self.medianCharSize * 0.4)  or y_diff < self.medianCharSize * 0.2)))      \
                                                                                                                        \
                        or (character.box.intersectY(neighbour.box) > 0 and character.box.intersectX(neighbour.box) > 0.9 * min(char_w, neighbour_w)
                            and min(char_aspect, neighbour_aspect) > 0.8 and abs(char_w - neighbour_w) < self.medianCharSize * 0.2 and max(char_w, neighbour_w) < self.medianCharSize * 2):

                        character.nearestNeighbours.append(neighbour)
                        neighbour.nearestNeighbours.append(character)

        for character in self.characters:
            if character.parentWord == None:
                if len(character.nearestNeighbours) >= 0:
                    word = Word([character])
                    words.append(word)

        return words

    def paint(self, image, color=colors.BLUE):

        for character in self.characters:
            image = character.paint(image, color)    # draw a dot at the word's center of mass.

        return image

class Word:

    def __init__(self, characters=[]):

        self.characters = set(characters)

        for character in characters:
            character.assignParentWord(self)

    def registerChildCharacter(self, character):

        if character not in self.characters:
            self.characters.add(character)

    def getBoundingBox(self):

        x_pos, y_pos = [], []
        for character in self.characters:
            for i in [0, 2]:
                x_pos += [character.box[i]]
                x_pos += [neighbour.box[i] for neighbour in character.nearestNeighbours]

                y_pos += [character.box[i + 1]]
                y_pos += [neighbour.box[i + 1] for neighbour in character.nearestNeighbours]

        pad = 2
        if len(self.characters) > 0:
            x_min, y_min = max(min(x_pos) - pad, 0), max(min(y_pos) - pad, 0)
            x_max, y_max = max(x_pos) + pad, max(y_pos) + pad
            return [x_min, y_min, x_max, y_max]

        return None

    def paint(self, image, color=colors.YELLOW):

        for character in self.characters:
            image = character.paint(image, color)

            for neighbour in character.nearestNeighbours:
                line = g.Line([character, neighbour])
                image = line.paint(image, color, thickness=1)

        box = self.getBoundingBox()
        if box is not None:
            cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 1)

        return image

    def getMask(self):

        box = self.getBoundingBox()
        h, w = box[3] - box[1], box[2] - box[0]
        mask = numpy.zeros((h, w), dtype='uint8')
        char_h = []

        for character in self.characters:
            char_h.append(character.box.height())

        median_char_h = numpy.median(char_h)

        for character in self.characters:
            x1, y1, x2, y2 = character.box
            x1, y1, x2, y2 = x1 - box[0], y1 - box[1], x2 - box[0], y2 - box[1]
            char_w = x2 - x1

            if y2 - y1 < char_w * 0.25 and y1 - box[1] > median_char_h * 0.2 :
                y1 = int(max(0, y1 - char_w * 0.25))
                y2 = int(y2 + char_w * 0.25)
            else:
                if y2 - y1 > median_char_h * 1.2:
                    y1 = int(y1 + char_w * 0.2)
                    y2 = int(max(0, y2 - char_w * 0.2))

            mask[y1:y2, x1:x2].fill(255)
            char_h.append(y2 - y1)

        # size = max(char_h)
        # ele = numpy.ones((3 * size, 3 * size))
        # mask = binary_closing(mask, ele)

        return mask


