import shutil
import logging
import math
import abc

import numpy as np

from ovnImage import check_dir
from glob import glob

from .MaskFeatures import MaskFeatures
from .Config import config
from .Image import ImageData


class DataSet:
    def __init__(self, path_data, prefix, max_images=None):
        self.path_data = path_data
        self.prefix = prefix

        self.n_samples = 0
        self.n_files = 0
        self.n_images = 0

        # Maximum number of images in the dataset
        self.max_images = max_images

    def files(self):
        return glob(self.path_data + self.prefix + "_*.npy")

    def isGenerated(self) -> bool:
        return len(self.files()) != 0

    def is_last_file(self, idx_current_image: int, len_files: int, max_files: int) -> bool:
        """
        Check if the current file is the last.

        :param idx_current_image: int index of the current image that has to be processed
        :param len_files: int Total number of images
        :param max_files: int Maximum of data files that should have the dataset
        :return:
        """
        return (idx_current_image >= len_files or
                (max_files is not None and self.n_files >= max_files) or
                (self.max_images is not None and self.n_images >= self.max_images))


class DataGenerator(abc.ABC):

    TH_HARD_SAMPLE = 0.9

    CLASS_IMAGE = ImageData

    def __init__(self, path_data: str, classname: str, eph: tuple, mean: tuple, cov: np.ndarray,
                 max_samples_mask: int = None, n_samples_file=None, limit_files: int = None):
        """
        Class to generate train and validation data files.

        :param path_data: Dataset path
        :param classname: Name of the target class/label
        :param eph: Ephsilon (y, x) class constant
        :param mean: Mean (y, x) of the train set target masks bounding box
        :param cov: Covariance matrices (y, x) of the train set target masks bounding box
        :param max_samples_mask: Maximum number of samples to take for each class in each image
        :param n_samples_file: Number of samples inside each built file
        :param limit_files: Maximum number of files to create for each data set
        """
        if max_samples_mask is None:
            max_samples_mask = config.DATA_MAX_PIXELS_CLASS

        if n_samples_file is None:
            n_samples_file = config.MAX_SAMPLES_FILE

        if limit_files is None:
            limit_files = config.MAX_TRAIN_FILES

        self.path_data = path_data
        self.max_samples_mask = max_samples_mask
        self.n_samples_file = n_samples_file
        self.limit_files = limit_files

        self.trainSet = DataSet(path_data, "train")
        self.testSet = DataSet(path_data, "test")

        self.mean = mean
        self.cov = cov
        self.ephsilon = eph

        self.f = MaskFeatures(self.ephsilon, self.mean, self.cov)

        self.classname = classname
        self.ids = []
        self.save_ids = False

    def data_generation(self, list_train_files, list_test_files, n_train_files=None, n_test_files=None,
                        max_train_files=None, max_test_files=None):
        # remove old data files
        shutil.rmtree(self.path_data)
        check_dir(self.path_data)

        # Set the maximum number of data files to create
        max_train_files = max_train_files if max_train_files is not None else self.limit_files
        max_test_files = max_test_files if max_test_files is not None else self.limit_files

        # Generate the data
        self.gen_files(list_train_files, "train", self.trainSet, n_train_files, max_train_files)
        self.gen_files(list_test_files, "test", self.testSet, n_test_files, max_test_files, class_weight='random')

    def gen_files(self, list_images, pref_files, set_data, n_files, max_files=None, class_weight="balance"):
        # if n_files is not set and class_weight is balanced compute the number of needed files
        if n_files is None and class_weight == 'balance':
            n_total_samples = len(list_images) * self.max_samples_mask * 2
            n_files = np.max([n_total_samples // self.n_samples_file, 1])

        if n_files is None:
            # create full data files until finish the available images
            set_data.n_images = 0
            while not set_data.is_last_file(set_data.n_images, len(list_images), max_files):
                set_data.n_files += 1
                logging.info("Creating " + pref_files + " dataset: " + str(set_data.n_files))
                data, n_imgs, n_samples = self._generate_data(list_images[set_data.n_images:], set_data, class_weight)
                set_data.n_samples += n_samples
                set_data.n_images += n_imgs
                np.save(self.path_data + pref_files + "_" + str(set_data.n_files) + ".npy", data)

        else:
            # Assign a balanced number of images to each data file
            images_files = np.array_split(list_images, n_files)
            for list_images_file in images_files:
                set_data.n_files += 1
                logging.info("Creating " + pref_files + " dataset: " + str(set_data.n_files))
                data, n_imgs, n_samples = self._generate_data(list_images_file, set_data, class_weight)
                set_data.n_samples += n_samples
                set_data.n_images += n_imgs
                np.save(self.path_data + pref_files + "_" + str(set_data.n_files) + ".npy", data)

    def _generate_data(self, images_path: list, set_data: DataSet, class_weight: str = "balance"):
        """
        Generate a matrix with data samples from images in images_path.

        :param images_path: list of strings -> paths of images
        :param set_data:
        :param class_weight: Distribution of the class samples of each image
        :return:
        """
        # initialize data structures
        pixels_wrote = np.zeros(self.CLASS_IMAGE.N_TARGETS + 1, dtype=np.uint32)
        data = np.zeros((self.n_samples_file, config.N_FEATURES + 1), dtype=np.float64)
        self.ids.append([])

        # iterate while there are images to process and data matrix is not full
        n_img = 0
        while (pixels_wrote.sum() < self.n_samples_file and
               n_img < len(images_path) and
               (set_data.max_images is None or n_img+set_data.n_images < set_data.max_images)):

            image = self.CLASS_IMAGE(images_path[n_img])

            im_gt = image.ground_truth_resized[self.classname]
            if im_gt is not None:
                # Format image and fill data with its samples
                indexs = self.__generate_class_data(data, image.im_resized, im_gt != 0, pixels_wrote,
                                                    class_weight=class_weight)
                if self.save_ids:
                    ids = self.get_ids(indexs, n_img)
                    self.ids[-1].extend(ids)

                logging.debug("Writted " + str(pixels_wrote))

            n_img += 1

        logging.debug("File has " + str(n_img) + " images")

        # Remove the not filled data rows
        n_samples = pixels_wrote.sum()
        if n_samples < self.n_samples_file:
            logging.info("Cutting matrix to " + str(n_samples) + " position")
            # cut matrix and exit
            data = np.delete(data, np.s_[n_samples:], 0)

        return data, n_img, n_samples

    def __generate_class_data(self, data: np.ndarray, input: np.ndarray, label: np.ndarray,
                              pixels_wrote, class_weight="balance"):
        """
        Fill the data matrix with samples of mask and its true labels from label.

        :param data: Matrix to fill
        :param input: Target image
        :param label: Image Ground truth
        :param pixels_wrote: Counter of the samples wrote for each class
        :return:
        """

        # Get the same number of pixels for each class
        if class_weight == 'balance':
            y_target, x_target = np.nonzero(label)
            y_others, x_others = np.nonzero(~label)

            # Ensure taking the same number of pixels of each class
            n_pixels = np.min([self.max_samples_mask, len(y_target), len(y_others)])

            # Select the sample pixels
            i = np.arange(len(y_target))
            np.random.shuffle(i)
            i = i[:n_pixels]
            i_target = (y_target[i], x_target[i])

            i = np.arange(len(y_others))
            np.random.shuffle(i)
            i = i[:n_pixels]
            i_others = (y_others[i], x_others[i])

            indexs = tuple(np.hstack((i_target, i_others)))

            pixels2write = n_pixels * 2

        # Get the samples of all the image pixels
        elif class_weight == 'all':
            indexs = np.indices(label.shape).reshape(-1, label.size)
            positions, feats = self.f.get_image_features(input, indexs)
            labels = (label != 0).flatten().astype(np.uint8)
            if int(pixels_wrote.sum() + label.size) >= data.shape[0]:
                n_data = int(data.shape[0] - pixels_wrote.sum())
                feats = feats[:n_data, :]
                labels = labels[:n_data]
            data[pixels_wrote.sum():int(pixels_wrote.sum() + label.size), :-1] = feats
            data[pixels_wrote.sum():int(pixels_wrote.sum() + label.size), -1] = labels
            pixels_wrote[0] += labels.size - np.count_nonzero(labels)
            pixels_wrote[1] += np.count_nonzero(labels)
            return indexs

        # Get random samples of the images, independently of its labels
        elif class_weight == 'random':
            n_pixels = self.max_samples_mask * 4

            i = np.arange(label.size)
            np.random.shuffle(i)
            i = i[:n_pixels]

            indexs = np.indices(label.shape).reshape(-1, label.size)
            indexs = (indexs[0][i], indexs[1][i])

            pixels2write = n_pixels

        else:
            raise Exception("Unknown class weight")

        # Compute number of samples to register
        pixels2write = pixels2write if pixels_wrote.sum() + pixels2write < data.shape[0] else int(data.shape[0] - pixels_wrote.sum())
        indexs = np.array([indexs[0][:pixels2write], indexs[1][:pixels2write]])

        # Compute samples data
        positions, feats = self.f.get_image_features(input, indexs)
        labels = (label[tuple(indexs)] != 0).astype(np.uint8)

        # Write data matrix
        ini_row = int(pixels_wrote.sum())
        data[ini_row:ini_row + pixels2write, :-1] = feats
        data[ini_row:ini_row + pixels2write, -1] = labels

        # Register number of wrote samples
        n_positives = np.count_nonzero(labels)
        pixels_wrote[0] += pixels2write - n_positives
        pixels_wrote[1] += n_positives

        return indexs

    def hardNegativesFileName(self, nfile):
        return self.path_data + "hn" + "_" + self.classname + "_" + str(nfile) + ".npy"

    def hardNegativesGeneration(self, clf, images_paths, perc_hn=0.5):
        logging.info("Hard negative mining ... ")
        max_img_samples = 1000
        i_data = 0
        max_hn = max_img_samples * len(images_paths)

        n_samples_needed = 0
        for train_file in self.trainSet.files():
            train_data = np.load(train_file)
            N = np.count_nonzero(train_data[:, -1] == 0)
            n_samples_needed += int(N * perc_hn)
        n_imgs = n_samples_needed // max_img_samples

        logging.debug("Using %i images", n_imgs)
        f = MaskFeatures(self.ephsilon, self.mean, self.cov)
        data = np.zeros((max_hn, config.N_FEATURES), dtype=np.float64)

        for i, path_im_x in enumerate(images_paths):
            image = self.CLASS_IMAGE(path_im_x)
            logging.info("Computing hard negatives from: " + image.im_name)

            im_gt = image.ground_truth_resized[self.classname]
            if im_gt is None:
                continue
            im_x = image.im_resized

            proba_mask = image.contruct_proba_img(clf, f, self.classname)

            y_true = (im_gt != 0).flatten()
            y_pred = (proba_mask > self.TH_HARD_SAMPLE).flatten()

            fp = ~y_true & y_pred
            n_fp = np.count_nonzero(fp)
            if n_fp == 0:
                continue
            n_samples = np.min([n_fp, max_img_samples])
            if i_data + n_samples > n_samples_needed:
                n_samples = n_samples_needed - i_data

            indexs = np.indices(im_x.shape).reshape(-1, im_x.shape[0] * im_x.shape[1])
            positions = indexs[:, fp]
            np.random.shuffle(positions.T)
            _, features = f.get_image_features(im_x, positions[:, :n_samples])
            data[i_data:i_data + n_samples, :] = features
            i_data = i_data + n_samples
            if i_data >= n_samples_needed:
                break

        if i_data < max_hn:
            logging.debug("Cutting matrix to " + str(i_data) + " position")
            # cut matrix and exit
            data = np.delete(data, np.s_[i_data:], 0)

        n_file = 0
        np.save(self.hardNegativesFileName(n_file), data)

    def HardNegativesAppend(self, perc_hn=0.5):
        n_file = 0
        hn_set = np.load(self.hardNegativesFileName(n_file))
        i_hn = 0
        for train_file in self.trainSet.files():
            logging.info("rebuilding: %s ", train_file)
            train_data = np.load(train_file)

            N = np.count_nonzero(train_data[:, -1] == 0)
            n2add = int(perc_hn * N)
            if i_hn + n2add > len(hn_set[:, -1]):
                n2add = len(hn_set[:, -1]) - i_hn

            data_add = hn_set[i_hn:i_hn + n2add, :]
            data_add = np.c_[data_add, np.zeros(data_add.shape[0], dtype=data_add.dtype)]

            new_dataset = np.r_[train_data, data_add]
            i_hn += n2add

            np.save(train_file, new_dataset)
            if i_hn > len(hn_set[:, -1]):
                break

    def hardNegativesInclusion(self, clfwrap, perc_hn=0.5):
        n_file = 0
        hn_set = np.load(self.hardNegativesFileName(n_file))
        i_hn = 0
        for train_file in self.trainSet.files():
            logging.info("rebuilding: %s ", train_file)

            train_data = np.load(train_file)
            y_true = train_data[:, -1]
            y_pred = clfwrap.clf.predict_proba(train_data[:, :-1])[:, 1]

            tn = (y_true == 0) & (y_pred < (1 - self.TH_HARD_SAMPLE))
            idxs_tn = tn.nonzero()

            N = np.count_nonzero(train_data[:, -1] == 0)
            n2add = int(perc_hn * N)
            if i_hn + n2add > len(hn_set[:, -1]):
                n2add = len(hn_set[:, -1]) - i_hn

            if n2add > len(idxs_tn[0]):
                n2add = len(idxs_tn[0])

            idxs_tn = list(idxs_tn[0][:n2add])
            train_data[idxs_tn, :-1] = hn_set[i_hn:i_hn + n2add, :]
            i_hn += n2add

            np.save(train_file, train_data)
            if i_hn > len(hn_set[:, -1]):
                break

    @classmethod
    def get_ids(cls, indexs, n_img):
        h, w = cls.CLASS_IMAGE.RESIZE_PARAMS['dsize']
        id_img = n_img * (10 ** int(math.log10(h*w)+3))
        ids = (indexs[0] * w) + indexs[1]
        ids += id_img
        return ids

    @classmethod
    def get_positions(cls, ids, n_img):
        h, w = cls.CLASS_IMAGE.RESIZE_PARAMS['dsize']
        id_img = n_img * (10 ** int(math.log10(h * w) + 3))
        ids -= id_img
        y = ids // w
        x = ids - y * w
        return y, x