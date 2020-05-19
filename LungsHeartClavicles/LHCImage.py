import cv2
import numpy as np

from shapemodels.Image import ImageData


def readGifImage(path):
    gif = cv2.VideoCapture(path)
    ret, frame = gif.read()
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


class LHCImage(ImageData):
    N_TARGETS = 1

    PATH_DATASET = ''
    PATH_DATASET_INPUT = PATH_DATASET + 'All247images/'
    PATH_DATASET_GT = PATH_DATASET + 'masks/'

    RESIZE_PARAMS = dict(dsize=(256, 256), fx=0, fy=0)

    CLASSES = ['lungs', 'clavicles', 'heart']
    NUM_GT = [2, 2, 1]
    ADD_HN = [
        True,
        True,
        True
    ]

    rows = 2048
    cols = 2048

    def _load_from_path(self, path) -> tuple:
        fd = open(path, 'rb')
        f = np.fromfile(fd, dtype=np.int16, count=self.rows * self.cols)
        fd.close()

        im_input = f.byteswap().astype(np.float64)
        im_input = im_input.reshape((self.rows, self.cols))
        return im_input, None

    @property
    def ground_truth(self):
        leftLung = readGifImage(self.PATH_DATASET_GT + "leftLung/" + self.im_name + ".gif")
        rightLung = readGifImage(self.PATH_DATASET_GT + "rightLung/" + self.im_name + ".gif")
        lungs_gt = cv2.bitwise_or(leftLung, rightLung)
        lungs_gt = cv2.resize(lungs_gt, self.im_input.shape)

        lefClavicle = readGifImage(self.PATH_DATASET_GT + "left clavicle/" + self.im_name + ".gif")
        rightClavicle = readGifImage(self.PATH_DATASET_GT + "right clavicle/" + self.im_name + ".gif")
        clavicles_gt = cv2.bitwise_or(lefClavicle, rightClavicle)
        clavicles_gt = cv2.resize(clavicles_gt, self.im_input.shape)

        heart_gt = readGifImage(self.PATH_DATASET_GT + "heart/" + self.im_name + ".gif")
        heart_gt = cv2.resize(heart_gt, self.im_input.shape)

        return {
            'lungs': lungs_gt,
            'clavicles': clavicles_gt,
            'heart': heart_gt
        }

    @property
    def ground_truth_items(self):
        leftLung = readGifImage(self.PATH_DATASET_GT + "leftLung/" + self.im_name + ".gif")
        leftLung = cv2.resize(leftLung, self.im_input.shape)

        rightLung = readGifImage(self.PATH_DATASET_GT + "rightLung/" + self.im_name + ".gif")
        rightLung = cv2.resize(rightLung, self.im_input.shape)

        lefClavicle = readGifImage(self.PATH_DATASET_GT + "left clavicle/" + self.im_name + ".gif")
        lefClavicle = cv2.resize(lefClavicle, self.im_input.shape)

        rightClavicle = readGifImage(self.PATH_DATASET_GT + "right clavicle/" + self.im_name + ".gif")
        rightClavicle = cv2.resize(rightClavicle, self.im_input.shape)

        heart_gt = readGifImage(self.PATH_DATASET_GT + "heart/" + self.im_name + ".gif")
        heart_gt = cv2.resize(heart_gt, self.im_input.shape)

        return {
            'lungs': [leftLung, rightLung],
            'clavicles': [lefClavicle, rightClavicle],
            'heart': [heart_gt]
        }
