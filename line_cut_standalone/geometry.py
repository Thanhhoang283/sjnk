import cv2
import numpy
import math

import colors
from pyqtree import Index

class Angle:

    def __init__(self, guess=None, degrees=None, radians=None, gradient=None):

        self.canonical = None # radians is the 'canonical' representation.

        if guess is not None:
            try:
                #try treating it like an Angle object
                self.radians(guess.radians())
            except:
                # otherwise treat it like a number in degrees
                self.degrees(guess)
        elif radians is not None:
            self.radians(radians)
        elif degrees is not None:
            self.degrees(degrees)
        elif gradient is not None:
            self.gradient(gradient)
        else:
            raise TypeError('Angle() takes at least one argument')

    def radians(self, newVal=None):
        if newVal is not None:
            self.canonical = Angle.sanitize(newVal)
        else:
            return self.canonical

    def degrees(self, newVal=None):
        if newVal is not None:
            rads = math.radians(newVal)
            self.canonical = Angle.sanitize(rads)
        else:
            return math.degrees(self.canonical)

    def gradient(self, newVal=None):
        # gradient = rise / run = tan(radians)
        if newVal is not None:
            rads = math.atan(newVal)
            self.canonical = Angle.sanitize(rads)
        else:
            return math.tan(self.canonical)

    def __add__(self, other):
        other = Angle(other)    # voila, we can now do angle2 = angle1 + 45
        raw = self.radians() + other.radians()
        return Angle(radians=Angle.sanitize(raw))

    def __sub__(self, other):
        other = Angle(other)
        raw = self.radians() - other.radians()
        return Angle(radians=Angle.sanitize(raw))

    @staticmethod
    def sanitize(rads):

        rads = float(rads)

        # put it into the range -pi < x < pi, including accounting for wrap-around
        rads = ((rads + math.pi) % (2*math.pi)) - math.pi

        # Our angles are symmetric. 3*pi/4 is equivalent to -pi/4
        if rads > (math.pi/2):
            rads = rads - math.pi
        elif rads < (-math.pi/2):
            rads = rads + math.pi

        return rads

    @staticmethod
    def average(angles):
        # important: this doesn't do well with angles close to +-90 degrees. Even if they're clustered close
        # to one point, they'll be split into >90 degrees and < 90 degrees sets, and average to zero.
        # This comes from the fact that angles are actually angles (i.e. symmetric), not bearings (directions).

        sumOfRads = 0.0
        for angle in angles:
            sumOfRads += angle.radians()
        rawAverage = sumOfRads / len(angles)
        return Angle(radians=Angle.sanitize(rawAverage))

class PointArray:

    def __init__(self, points=[]):

        self.points = []
        for point in points:
            # make sure that each point is a Point instance. Also allows us to accept a generator.
            self.points.append(Point(point))

    def __str__(self):
        # human-readable output
        strings = [point.__str__() for point in self.points]

        return "[%s]" %(", ".join(strings))

    def __repr__(self):
        # machine-readable output
        return self.__str__()

    def append(self, point):
        self.points.append(Point(point))

    def numpyArray(self):
        return numpy.array([ [list(point.align())] for point in self.points ])

    def __getitem__(self, key):
        return self.points.__getitem__(key)

    def __setitem__(self, key, value):
        self.points.__setitem__(key, value)

    def __delattr__(self, key):
        self.points.__setitem__(key, None)

    def __reversed__(self):
        return self.points.__reversed__()

    def __len__(self):
        return self.points.__len__()

    def __iter__(self):
        return self.points.__iter__()

    def paint(self, image, color):
        for point in  self.points:
            image = point.paint(image, color)
        return image


class Point:

    def __init__(self, foo=None, bar=None):

        try:
            # If foo is an array, use that and ignore bar.
            # Note that this also means that Point(Point(foo, bar)) is harmless
            self.x = foo[0]
            self.y = foo[1]
        except:
            # Otherwise treat foo and bar like two numbers
            self.x = foo
            self.y = bar

        self.isPoint = True     # used to test instance type.

    def align(self):
        # return a new point instance where .x and .y are integers

        return Point(numpy.int0(numpy.around([self.x, self.y])))

    def cv2point(self):

        return tuple(self.align())

    def rotate(self, angle):

        angle = Angle(angle)
        rotatedPoint = Point()
        rotatedPoint.x = self.x*math.cos(-angle.radians()) - self.y*math.sin(-angle.radians())
        rotatedPoint.y = self.x*math.sin(-angle.radians()) + self.y*math.cos(-angle.radians())

        return rotatedPoint

    def __str__(self):
        # human-readable output
        return "(x:%s, y:%s)" %(self.x, self.y)

    def __repr__(self):
        # machine-readable output
        return self.__str__()

    def __getitem__(self, key):
        # this is a hack that allows the object to be treated like a list.
        return [self.x, self.y].__getitem__(key)

    def __setitem__(self, key, value):
        if key == 0:
            self.x = value
        elif key == 1:
            self.y = value
        else:
            raise KeyError('key must be 0 or 1')

    def __delattr__(self, key):
        self.__setitem__(key, None)

    def __reversed__(self):
        return Point(self.y, self.x)

    def __len__(self):
        return 2

    def __iter__(self):

        yield self.x
        yield self.y
        raise StopIteration

    def __add__(self, other):
        result = Point()
        result.x = self.x + other.x
        result.y = self.y + other.y
        return result

    def __sub__(self, other):
        result = Point()
        result.x = self.x - other.x
        result.y = self.y - other.y
        return result

    def paint(self, image, color, diameter=1):
        cv2.circle(image, self.cv2point(), diameter, color, 3, cv2.LINE_AA)
        return image

    @staticmethod
    def distance(start, end):

        start = Point(start)
        end = Point(end)

        delta = end - start

        distance = math.sqrt(delta.x**2 + delta.y**2)

        return distance

    @staticmethod
    def midpoint(start, end):

        start = Point(start)
        end = Point(end)

        midpoint = Point()
        midpoint.x = float(start.x + end.y) / 2
        midpoint.y = float(start.x + end.y) / 2

        return midpoint


class Line:

    def __init__(self, points=[], inputAngle=None, frame=None):

        self.frame = frame

        self.start = None
        self.end = None
        self.angle = None

        if inputAngle != None:
            inputAngle = Angle(inputAngle)
        self.inputAngle = inputAngle

        self.points = PointArray(points)
        self.update()

    def append(self, point):

        self.points.append(point)
        self.update()

    def intersect(self, other):

        if (self.start is None) or (self.end is None):
            raise Exception('The PixelLine is underspecified; it requires at least two points')
        if (other.start is None) or (other.end is None):
            raise Exception('The PixelLine is underspecified; it requires at least two points')

        otherX = float(other.start.x)
        otherY = float(other.start.y)
        otherM = float(other.angle.gradient())

        selfX = float(self.start.x)
        selfY = float(self.start.y)
        selfM = float(self.angle.gradient())

        point = Point()
        point.x = (otherY - selfY + selfM*selfX - otherM*otherX) / (selfM - otherM)
        point.y = selfY + selfM*(point.x - selfX)

        return point

    def update(self):

        if (self.inputAngle is not None) and (len(self.points) >= 1):
            self.lineFromPointAngle()

        elif len(self.points) < 2:
            self.start = None
            self.end = None
            self.angle = None

        elif len(self.points) == 2:
            self.lineFromTwoPoints()

        else:
            self.leastSquaresLine()

        self.clipToFrame()

    def lineFromPointAngle(self):
        # We find the line based on the angle and the first point. Note that in this case, the line
        # is effectively infinite.

        hypotenuse = 4000
        datum = self.points[0]
        angle = self.inputAngle + 90

        offset = Point()
        offset.x = int(hypotenuse * math.cos(angle.radians()))
        offset.y = int(hypotenuse * math.sin(angle.radians()))

        self.start = datum - offset
        self.end = datum + offset
        self.angle = self.inputAngle

    def lineFromTwoPoints(self):
        # This is the only case in which the line has a visible start and end point.

        self.start = self.points[0]
        self.end = self.points[1]
        self.angle = self.calculateAngle(self.start, self.end)

    def leastSquaresLine(self):
        # try to fit a least-squares trend line

        multiplier = 2000
        dx, dy, x0, y0 = cv2.fitLine(self.points.numpyArray(), cv2.cv.CV_DIST_L2, 0, 0.01, 0.01)

        self.start = Point(int(x0 - dx*multiplier), int(y0 - dy*multiplier))
        self.end = Point(int(x0 + dx*multiplier), int(y0 + dy*multiplier))
        self.angle = self.calculateAngle(self.start, self.end)

    def calculateAngle(self, start, end):

        rise = float(self.end.y) - float(self.start.y)
        run = float(self.end.x) - float(self.start.x)

        return Angle(radians=math.atan2(rise, run))

    def clipToFrame(self):
        if self.frame is not None:
            rawStart, rawEnd = cv2.clipLine(self.frame, self.start, self.end)
            self.start = Point(rawStart)
            self.end = Point(rawEnd)

    def paint(self, image, color=colors.BLUE, thickness = 4):

        if (self.start is None) or (self.end is None):
            raise Exception('The Line is underspecified; it requires at least two points')
        else:
            cv2.line(image, self.start.cv2point(), self.end.cv2point(), color, thickness, cv2.LINE_AA)

        return image


class Rect:

    def __init__(self, top_left, bottom_right):
        self.top_left = top_left
        self.bottom_right = bottom_right

        self.left, self.top = top_left
        self.right, self.bottom = bottom_right

    def width(self):
        return self.right - self.left

    def height(self):
        return self.bottom - self.top

    def area(self):
        return 1.0 * self.width() * self.height()

    def aspect(self):
        return 1.0 * self.width() / self.height()

    def intersectArea(self, rect):

        x1, y1, x2, y2 = self
        x3, y3, x4, y4 = rect

        left, right = max(x1, x3), min(x2, x4)
        top, bottom = max(y1, y3), min(y2, y4)

        if left < right - 0 and top < bottom - 2:
            return 1.0 * (right - left) * (bottom - top)

        return 0.0

    def intersectX(self, rect, pad = 5):
        x1, y1, x2, y2 = self
        x3, y3, x4, y4 = rect

        left, right = max(x1, x3), min(x2, x4)
        if left < right + pad:
            return right + pad - left
        return 0

    def intersectY(self, rect, pad = 0):
        x1, y1, x2, y2 = self
        x3, y3, x4, y4 = rect

        top, bottom = max(y1, y3), min(y2, y4)
        if top < bottom + pad:
            return bottom + pad - top
        return 0

    def as_list(self):
        return [self.left, self.top, self.right, self.bottom]

    def slice(self):
        return slice(self.top, self.bottom), slice(self.left, self.right)

    def is_overlap_by(self, other, pad=3):
        x1, y1, x2, y2 = self
        x3, y3, x4, y4 = other
        x3 -= pad
        y3 -= pad
        x4 += pad
        y4 += pad
        return (x1 >= x3 and x2 <= x4 and y1 >= y3 and y2 <= y4)

    def __iter__(self):
        yield self.left
        yield self.top
        yield self.right
        yield self.bottom
        raise StopIteration

    def __getitem__(self, item):
        return self.as_list()[item]

    def __str__(self):
        # human-readable output
        strings = [str(p) for p in self]

        return "[%s]" %(", ".join(strings))

    def pad(self, pad_x, pad_y):
        return Rect((self.left - pad_x, self.top - pad_y), (self.right + pad_x, self.bottom + pad_y))

    def __add__(self, other):
        if other is None:
            return self

        x1, y1, x2, y2 = self
        x3, y3, x4, y4 = other

        left, right = min(x1, x3), max(x2, x4)
        top, bottom = min(y1, y3), max(y2, y4)

        return Rect((left, top), (right, bottom))

def union_rects(rects):
    if len(rects) == 0:
        return None

    r = rects[0]
    for i in range(1, len(rects)):
        r += rects[i]
    return r

def union_words(word_a, word_b):

    rect_a, mask_a = word_a
    rect_b, mask_b = word_b

    if rect_b is None or mask_b is None:
        return word_a

    merged_rect = rect_a + rect_b

    new_mask = numpy.zeros((merged_rect.height(), merged_rect.width()), dtype='uint8')
    new_mask[rect_a.top - merged_rect.top : rect_a.bottom - merged_rect.top, rect_a.left - merged_rect.left : rect_a.right - merged_rect.left][mask_a > 0] = 255
    new_mask[rect_b.top - merged_rect.top : rect_b.bottom - merged_rect.top, rect_b.left - merged_rect.left : rect_b.right - merged_rect.left][mask_b > 0] = 255

    return (merged_rect, new_mask)

def merge_rect(rects, masks, i, merged_with, done):
    if done[i]:
        return None, None

    done[i] = True
    r = rects[i]
    mask = masks[i]

    for j in merged_with[i]:
        if not done[j]:
            new_words = merge_rect(rects, masks, j, merged_with, done)
            r, mask = union_words((r, mask), new_words)

    return r, mask

def expand_char_mask(words, thres):
    for i, word in enumerate(words):
        r, m = word

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (int(thres * 15), int(thres * 0.6)))
        kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (int(thres * 15), int(thres * 1.3)))

        m = cv2.dilate(m, kernel)
        m = cv2.erode(m, kernel2)

        words[i] = (r, m)

def filter_and_merge_overlap_boxes(words, thres, maxsize, use_merge_same_line_only = False, same_line_multiplier = 1.0):

    if not use_merge_same_line_only:
        words_len = [l for _, _, l in words]
        words = [(b, m) for b, m, _ in words]

    if (len(words) < 2):
        if use_merge_same_line_only:
            expand_char_mask(words, thres)
        return words

    # create quad tree
    h, w = maxsize
    spindex = Index(bbox=(0, 0, w - 1, h - 1))

    rects = [Rect((b[0], b[1]), (b[2], b[3])) for b, _ in words]
    for i, r in enumerate(rects):
        spindex.insert(i, r.as_list())

    masks = [m for _, m in words]
    is_slant = [False] * len(rects)

    for i, mask in enumerate(masks):
        mask = (mask > 0)
        proj_x = numpy.sum(mask, axis=0)

        if (mask.shape[0] > thres * 5 or mask.shape[0] < thres * 2) and mask.shape[1] > thres * 20 and numpy.percentile(proj_x, 90) < mask.shape[0] * 0.7:
            is_slant[i] = True
        else:
            masks[i].fill(0)
            mh, mw = masks[i].shape
            masks[i][int(mh * 0.12) : int(mh * 0.88), :].fill(255)

    merged_with = [set() for i in range(len(rects))]

    done = [False] * len(rects)
    new_boxes = []

    height_diff_thres = thres

    for i in range(len(rects)):
        same_line_thres = max(int(thres * 0.8), 5) if not use_merge_same_line_only else max(
                                   rects[i].height() * 0.3, thres * same_line_multiplier)
        overlapbbox = rects[i].pad(same_line_thres, 2).as_list()
        matches = spindex.intersect(overlapbbox)

        for j in matches:

            if (i == j) or is_slant[i] or is_slant[j]: continue

            same_line_thres = max(int(thres * 0.8), 5) if not use_merge_same_line_only else max(
                min(rects[i].height(), rects[j].height()) * 0.3, thres * same_line_multiplier)

            #if use_merge_same_line_only: height_diff_thres = max(thres, min(rects[i].height(), rects[j].height()) * 0.7)

            rule_intersect_boxes_on_same_line = rects[i].intersectY(rects[j]) > 0.8 * min(rects[i].height(), rects[j].height())  \
                                                        and rects[i].intersectX(rects[j], pad = same_line_thres) > 0 \
                                                        and rects[i].intersectX(rects[j], pad=0) < thres * 20

            rule_big_intersect_area = rects[i].intersectArea(rects[j]) > 0.33 * min(rects[i].area(), rects[j].area()) \
                                      and max(rects[i].width(), rects[j].width()) < thres * 35 \
                                      and min(rects[i].width(), rects[j].width()) < thres * 8
                                      #and min(rects[i].height(), rects[j].height()) > thres * 10

            rule_same_height_or_very_short_box = abs(rects[i].height() - rects[j].height()) < height_diff_thres * 0.8           \
                                                or (max(rects[i].height(), rects[j].height()) < thres * 1.8
                                                        and min(rects[i].width(), rects[j].width()) < thres * 9) \
                                                 or (abs(rects[i].height() - rects[j].height()) < thres * 1.6
                                                    and max(rects[i].height(), rects[j].height()) < thres * 3)

            rule_small_diacritic_inline = rects[i].intersectY(rects[j]) > 0.94 * rects[i].height() \
                                              and rects[i].intersectX(rects[j], pad=max(int(thres * 0.8), 5)) > 0  and (
                                              rects[i].height() < rects[j].height() * 0.4 and
                                              rects[i].width() < thres * 9) and rects[j].height() < thres * 15 and rects[j].width() > thres * 10

            if not use_merge_same_line_only:

                rule_same_width_box_intersect = rects[i].intersectX(rects[j], pad=0) > 0.9 * min(rects[i].width(),
                                                                                                 rects[j].width()) \
                                                and rects[i].intersectY(rects[j],
                                                                        pad=2) > 0 and abs(
                                                rects[i].width() - rects[j].width()) < thres \
                                                and max(rects[i].width(), rects[j].width()) < thres * 12 and min(
                                                rects[i].height(), rects[j].height()) < thres * 5  \
                                                and rects[i].height() + rects[j].height() < max(rects[i].width(), rects[j].width()) * 1.3 \
                                                and words_len[i] == 1 and words_len[j] == 1

                rule_very_big_intersect_with_height_limit = ((rects[i].intersectY(rects[j]) > 0.6 * max(rects[i].height(), rects[j].height())
                            and rects[i].intersectX(rects[j], pad = 0) > thres * 1.6)
                            or (rects[i].intersectX(rects[j]) > 0.4 * max(rects[i].width(), rects[j].width())
                                and rects[i].intersectY(rects[j]) > thres * 1.4)) and max(rects[i].height(), rects[j].height()) < thres * 3.7  \
                            and max(rects[i].width(), rects[j].width()) < thres * 30
            else:

                rule_very_big_intersect_with_height_limit = False
                rule_same_width_box_intersect = False


            if ((rule_intersect_boxes_on_same_line
                        or rule_big_intersect_area)
                    and rule_same_height_or_very_short_box) \
                    or rule_same_width_box_intersect \
                    or rule_very_big_intersect_with_height_limit \
                    or rule_small_diacritic_inline:

                merged_with[i].add(j)
                merged_with[j].add(i)


    #print('merging')
    for i in range(len(rects)):
        if not done[i]:
            new_boxes.append(merge_rect(rects, masks, i, merged_with, done))

    ### filter overlap boxes:
    is_overlap = [False] * len(new_boxes)
    spindex = Index(bbox=(0, 0, w - 1, h - 1))
    for i in range(len(new_boxes)):
        spindex.insert(i, new_boxes[i][0].as_list())

    for i in range(len(new_boxes)):
        overlapbbox = new_boxes[i][0].pad(3, 3).as_list()
        matches = spindex.intersect(overlapbbox)
        for j in matches:
            if i == j: continue
            if new_boxes[i][0].is_overlap_by(new_boxes[j][0]):
                is_overlap[i] = True
                #print('overlap')
            # if new_boxes[j][0].is_overlap_by(new_boxes[i][0]):
            #     is_overlap[j] = True
            #     #print('overlap')

    if use_merge_same_line_only:
        expand_char_mask(new_boxes, thres)

    new_boxes = [(new_boxes[i][0].as_list(), new_boxes[i][1]) for i in range(len(new_boxes)) if not is_overlap[i]]

    return new_boxes







