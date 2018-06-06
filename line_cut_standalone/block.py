import cv2
import math
import numpy
import subprocess
import os

import colors
import geometry as g
import text
from dimension import Dimension
#from stopwatch import Stopwatch
import numpy

#stopwatch = Stopwatch()

class Block:

    def __init__(self, image, scale, showSteps=False):

        #stopwatch.reset(path)

        self.showSteps = showSteps
         #cv2.imread(path)
        if len(image.shape) > 2:
            greyscaleImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            greyscaleImage = image


        self.characters = text.CharacterSet(greyscaleImage, scale)
        self.words = self.characters.getWords()

        self.image = image

        #stopwatch.lap("finished analysing page")
        #stopwatch.endRun()


    def paint(self, image = None):

        #print('word', len(self.words))

        if image is None:
            image = 255 - self.characters.image
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        for word in self.words:
            image = word.paint(image, colors.RED)

        return image

    def save(self, path):

        image = self.image.copy()
        #image = text.threshold(image, cv2.THRESH_OTSU, method=cv2.THRESH_BINARY)

        image = self.paint(image)
        cv2.imwrite(path, image)

    def display(self, image, boundingBox=(800,800), title='Image'):

        #stopwatch.pause()

        if boundingBox:
            maxDimension = Dimension(boundingBox[0], boundingBox[1])
            displayDimension = Dimension(image.shape[1], image.shape[0])
            displayDimension.fitInside(maxDimension)
            print("Display Dimmension: ", tuple(displayDimension))
            image = cv2.resize(image, tuple(displayDimension))

        cv2.namedWindow(title, cv2.WINDOW_AUTOSIZE)
        cv2.imshow(title, image)
        cv2.waitKey()

        #stopwatch.unpause()

    def getWordBoundingBoxes(self):
        return [(w.getBoundingBox(), w.getMask(), len(w.characters)) for w in self.words]
        
    def show(self, boundingBox=None, title="Image"):    #textImage

        image = self.image.copy()
        image = self.paint(image)
        self.display(image, boundingBox, title)

    def extractWords(self, sourceImage):

        image = sourceImage.copy()
        # image = threshold(image)
        #
        # tempImageFile = os.path.join('src', 'tempImage.tiff')
        # tempTextFile = os.path.join('src', 'tempText')
        #
        # mask = numpy.zeros(image.shape, numpy.uint8)
        # singleWord = numpy.zeros(image.shape, numpy.uint8)
