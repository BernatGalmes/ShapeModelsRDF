import os
import cv2
import numpy as np
import logging
import matplotlib.pyplot as plt
import ovnImage as ovni
import abc

from sklearn.cluster import DBSCAN

from shapemodels.Config import config
from timeit import default_timer as timer


class ImageData(abc.ABC):
    RESIZE_PARAMS = dict(dsize=None, fx=0.5, fy=0.5)

    CLASSES = ['positives', 'negatives']

    # Maximum number of samples to predict together
    PREDICTION_STEP = 400000

    # Postprocessing variables
    OPENING_KERNEL = np.ones((5, 5), np.uint8)
    CLOSING_KERNEL = np.ones((5, 5), np.uint8)

    # Morphological operations to perform to resulting mask
    POST_OPERATIONS = ['closing', 'opening']

    VISUALIZATION_COLORS = [
        (0, 0, 255),
        (0, 255, 0),
        (255, 255, 0)
    ]

    """
    Class operations setters
    """
    @classmethod
    def set_clossing_kernel(cls, kernel):
        cls.CLOSING_KERNEL = kernel

    @classmethod
    def set_opening_kernel(cls, kernel):
        cls.OPENING_KERNEL = kernel

    @classmethod
    def set_post_operations(cls, value):
        cls.POST_OPERATIONS = value

    def __init__(self, path):
        """

        :param path:
        :param class_datagen:
        """

        self.im_name = str(os.path.basename(path).split('.')[0])

        # load data from file
        self.im_input, self.labels = self._load_from_path(path)
        self.im_resized = cv2.resize(self.im_input, **self.RESIZE_PARAMS)

        # Get the image new dimensions
        if self.RESIZE_PARAMS['dsize'] is None:
            self.im_h = int(self.im_input.shape[0] * self.RESIZE_PARAMS['fy'])
            self.im_w = int(self.im_input.shape[1] * self.RESIZE_PARAMS['fx'])
        else:
            self.im_w, self.im_h = self.RESIZE_PARAMS['dsize']

        # Get the image pixels index
        self.indexs = np.indices(self.im_resized.shape).reshape(-1, self.im_resized.shape[0] * self.im_resized.shape[1])

        self.proba_images = {}
        self.ims_clusters = {}
        self.masks_prob = {}
        self.classifiers = []
        self.clusters = {}
        self.clusters_data = {}
        self.bbs = {cl: [] for cl in self.CLASSES}
        self.ini_bbs = {cl: [] for cl in self.CLASSES}

    @abc.abstractmethod
    def _load_from_path(self, path) -> tuple:
        ...

    @property
    @abc.abstractmethod
    def ground_truth_items(self) -> dict:
        ...

    @property
    @abc.abstractmethod
    def ground_truth(self) -> dict:
        ...

    @property
    def ground_truth_items_resized(self) -> dict:
        gts = self.ground_truth_items
        for classname in self.CLASSES:
            for i, im_gt in enumerate(gts[classname]):
                if im_gt is not None:
                    gts[classname][i] = cv2.resize(im_gt, **self.RESIZE_PARAMS)

        return gts

    @property
    def ground_truth_resized(self) -> dict:
        gts = self.ground_truth
        for classname in self.CLASSES:
            if gts[classname] is not None:
                gts[classname] = cv2.resize(gts[classname], **self.RESIZE_PARAMS)

        return gts

    def get_ground_truths_bbs(self):
        ground_truths_bbs = {cl: [] for cl in self.CLASSES}
        for i_class, classname in enumerate(self.CLASSES):
            for im_gt in self.ground_truth_items_resized[classname]:
                im_gt = cv2.dilate(im_gt, np.ones((7, 7), dtype=np.uint8))
                bb_gt = np.array(ovni.masks.bounding_box(im_gt))
                if bb_gt[0] is not None:
                    ground_truths_bbs[classname].append((im_gt, bb_gt))
        return ground_truths_bbs

    def save_image_data(self, f, classname: str, filename: str):
        """
        Save image features as np.memmap.

        :param f: MaskFeatures Class feature extractor object instance
        :param classname: Predicting class name
        :param filename: File name where to store the np.memmap
        :return:
        """
        labels = (self.ground_truth_resized[classname] != 0).flatten().astype(np.uint8)

        data_image = np.memmap(filename, dtype=np.float64, shape=(self.im_resized.size, config.N_FEATURES + 1),
                               mode='w+')

        # if the image can be predicted in a single step not iterate
        if self.im_resized.size < self.PREDICTION_STEP:
            positions, features = f.get_image_features(self.im_resized, self.indexs)

            data_image[:, :-1] = features
            data_image[:, -1] = labels

        # else predict the image in steps
        else:
            for last_index in range(self.PREDICTION_STEP, self.im_resized.size, self.PREDICTION_STEP):
                ini_index = last_index - self.PREDICTION_STEP
                positions, features = f.get_image_features(self.im_resized, self.indexs[:, ini_index:last_index])

                data_image[ini_index:last_index, :-1] = features
                data_image[ini_index:last_index, -1] = labels[ini_index:last_index]

            # process remaining pixels
            last_index = self.im_resized.size - (self.im_resized.size % self.PREDICTION_STEP)
            if last_index < self.im_resized.size:
                positions, features = f.get_image_features(self.im_resized, self.indexs[:, last_index:])

                data_image[last_index:, :-1] = features
                data_image[last_index:, -1] = labels[last_index:]

    def contruct_proba_img(self, clf, f, classname):
        """
        Given a classifier and its feature extractor construct the probability map.

        :param clf: sklearn trained classifier
        :param f: Features extractor
        :param classname: Predicting class name
        :return:
        """
        proba_img = np.zeros((self.im_resized.shape[0], self.im_resized.shape[1]),
                             dtype=np.float32)

        # if the image can be predicted in a single step not iterate
        if proba_img.size < self.PREDICTION_STEP:
            positions, features = f.get_image_features(self.im_resized, self.indexs)
            predicted = clf.predict_proba(features)
            proba_img[tuple(positions)] = predicted[:, 1]

        # else predict the image in steps
        else:
            for last_index in range(self.PREDICTION_STEP, proba_img.size, self.PREDICTION_STEP):
                ini_index = last_index - self.PREDICTION_STEP
                positions, features = f.get_image_features(self.im_resized, self.indexs[:, ini_index:last_index])

                predicted = clf.predict_proba(features)
                proba_img[tuple(positions)] = predicted[:, 1]

            # process remaining pixels
            last_index = self.im_resized.size - (self.im_resized.size % self.PREDICTION_STEP)
            if last_index < proba_img.size:
                positions, features = f.get_image_features(self.im_resized, self.indexs[:, last_index:])

                predicted = clf.predict_proba(features)
                proba_img[tuple(positions)] = predicted[:, 1]

        self.proba_images[classname] = proba_img
        return proba_img

    def preprocess_prob(self, th: float, classname: str):
        """
        Compute the segmentation results.
        Threshold the probability map and processing it with
        some morphological operations.

        :param th: Threshold to apply to the probability map
        :param classname: Name of the class

        :return: The resulting binary image
        """
        mask = (self.proba_images[classname] > th).astype(np.uint8)
        for operation in self.POST_OPERATIONS:
            if operation == 'closing' and self.CLOSING_KERNEL is not None:
                mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.CLOSING_KERNEL)

            elif operation == 'opening' and self.OPENING_KERNEL is not None:
                mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.OPENING_KERNEL)

        self.masks_prob[classname] = mask
        return mask

    def clustering(self, classname):
        """
        Compute the segmentation results clusters.
        Identify each object instance onto the image.

        :param classname: Name of the class
        :return:
        """
        mask_proba = self.masks_prob[classname]

        positions = np.nonzero(mask_proba)
        self.clusters_data[classname] = np.array(positions).T
        if len(positions[0]) == 0:
            self.clusters[classname] = None
        else:
            self.clusters_data[classname] = np.array(positions).T
            cluster = DBSCAN(eps=7, metric='euclidean').fit(self.clusters_data[classname])
            self.clusters[classname] = cluster

    def get_bbs(self, bb_size, classname):
        """
        Get the Bounding boxes of each detected object instance.

        :param bb_size: Default bounding box size
        :param classname: Target Class name
        :return: tuple of initial and final bounding boxes list
        """
        cluster = self.clusters[classname]
        if cluster is None:
            return [], []

        positions = np.array(np.nonzero(self.masks_prob[classname]))
        labels = cluster.labels_

        bb_h, bb_w = int(bb_size[0]), int(bb_size[1])

        # Number of clusters in labels, ignoring noise if present.
        n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
        ini_bbs, bbs = [], []
        for k in range(n_clusters_):
            elements = positions[:, labels == k]

            # Bounding box parameters
            p_mid = np.median(np.unique(elements[0, :])), np.median(np.unique(elements[1, :]))
            y, x = int(p_mid[0] - bb_size[0] // 2), int(p_mid[1] - bb_size[1] // 2)

            # discart too small clusters
            cluster_h = np.ptp(elements[0, :])
            cluster_w = np.ptp(elements[1, :])
            if cluster_h < 0.1 * bb_h or cluster_w < 0.1 * bb_w:
                continue

            track_window = (x, y, bb_w, bb_h)
            ini_bbs.append(track_window)

            term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 100, 1)
            ret, track_window = cv2.CamShift(self.proba_images[classname], track_window, term_crit)
            # ret, track_window = cv2.meanShift(self.proba_images[classname], track_window, term_crit)

            bbs.append(track_window)
        return ini_bbs, bbs

    def process_image(self, classifiers, feat_extractors, ths, bb_sizes, timings):
        """
        Process the image with all steps and obtain the initial and final found
        bounding boxes
        :param classifiers:
        :param feat_extractors:
        :param ths:
        :param bb_sizes:
        :param timings:
        :return:
        """
        ini_bbs, bbs = {}, {}
        for i_class, classname in enumerate(self.CLASSES):
            logging.info("Processing image: " + self.im_name)
            ini_time = timer()
            # Construct prob image
            self.contruct_proba_img(classifiers[i_class].clf,
                                    feat_extractors[classname],
                                    classname)
            end_time_predict = timer()

            # Process prob image
            self.preprocess_prob(ths[classname][0],
                                 classname)
            end_time_processing = timer()

            # Cluster results
            self.clustering(classname)
            end_time_clustering = timer()

            # Get Bounding boxes
            ini_bbs[classname], bbs[classname] = self.get_bbs(bb_sizes[classname],
                                                              classname)
            end_time_bbs = timer()
            if timings is not None:
                timings['predict'].append(end_time_predict - ini_time)
                timings['process'].append(end_time_processing - end_time_predict)
                timings['clustering'].append(end_time_clustering - end_time_processing)
                timings['bbs'].append(end_time_bbs - end_time_clustering)
                timings['total'].append(end_time_bbs - ini_time)
                timings['class'].append(classname)
            logging.debug("Time processing image: %f", end_time_bbs - ini_time)
        return ini_bbs, bbs

    def build_cluster_image(self, cluster, cluster_data):
        im_clusters = np.zeros((self.im_h, self.im_w, 3), np.uint8)
        if cluster is not None:
            labels = cluster.labels_

            # Black removed and is used for noise instead.
            unique_labels = set(labels)

            colors2 = [plt.cm.Spectral(each)
                       for each in np.linspace(0, 1, len(unique_labels))]

            for k, col in zip(unique_labels, colors2):
                pixels_cluster = cluster_data[labels == k, :]
                im_clusters[pixels_cluster[:, 0], pixels_cluster[:, 1], :] = (np.array(col[:3])*255).astype(np.uint8)

        return im_clusters
