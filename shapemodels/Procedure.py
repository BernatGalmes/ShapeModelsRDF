import os
import cv2
import numpy as np
import pandas as pd
import ovnImage as ovni
import matplotlib.pyplot as plt
import tifffile

from timeit import default_timer as timer

import logging

from openpyxl import load_workbook

from shapemodels.Config import config
from shapemodels.MaskFeatures import MaskFeatures
from shapemodels.Image import ImageData
import shapemodels.helpers as helpers

from scipy.spatial import distance as dist


# Malisiewicz et al.
def non_max_suppression_fast(boxes, overlapThresh):
    # if there are no boxes, return an empty list
    if len(boxes) == 0:
        return []
    # if the bounding boxes integers, convert them to floats --
    # this is important since we'll be doing a bunch of divisions
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")
    # initialize the list of picked indexes
    pick = []
    # grab the coordinates of the bounding boxes
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = x1 + boxes[:, 2]
    y2 = y1 + boxes[:, 3]

    # compute the area of the bounding boxes and sort the bounding
    # boxes by the bottom-right y-coordinate of the bounding box
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)
    # keep looping while some indexes still remain in the indexes
    # list
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the
        # index value to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)
        # find the largest (x, y) coordinates for the start of
        # the bounding box and the smallest (x, y) coordinates
        # for the end of the bounding box
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])
        # compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        # compute the ratio of overlap
        overlap = (w * h) / area[idxs[:last]]
        # delete all indexes from the index list that have
        idxs = np.delete(idxs, np.concatenate(([last],
                                               np.where(overlap > overlapThresh)[0])))
    # return only the bounding boxes that were picked using the
    # integer data type
    return boxes[pick].astype("int")


def distance_boundig_boxes(bb1, bb2):
    x, y, w, h = bb1
    mid_point_1 = (y + (h // 2), x + (w // 2))

    x, y, w, h = bb2
    mid_point_2 = (y + (h // 2), x + (w // 2))

    return dist.euclidean(mid_point_1, mid_point_2)


class Procedure:


    @staticmethod
    def train_and_report(train_files, val_files, rdf_params, classname, file_clf, file_report, force_clf_creation):
        clfwrap = helpers.ClfWrap(file_clf, rdf_params, force_clf_creation=force_clf_creation,
                                  labels=["Other", classname])

        ini_time = timer()
        clfwrap.train(train_files, trees4file=config.rf_inc_trees_fit)
        end_time_train = timer()

        y_true, y_pred = clfwrap.predict(val_files)
        end_time_predict = timer()

        clfwrap.plot_report(y_true, y_pred, report_filename=file_report)
        metrs = ovni.metrics.get_classification_metrics(y_true, y_pred)
        metrs['n. train files'] = len(train_files)
        metrs['time train (s)'] = end_time_train - ini_time
        metrs['time predict (s)'] = end_time_predict - end_time_train
        return clfwrap, metrs

    @staticmethod
    def write_metrics(metrics, file_metrics, sheet_name):
        if len(metrics) > 0:
            metrics_df = pd.DataFrame(metrics)
            with pd.ExcelWriter(file_metrics) as writer:
                if os.path.isfile(file_metrics):
                    writer.book = load_workbook(file_metrics)
                metrics_df.to_excel(writer, sheet_name=sheet_name)

    @staticmethod
    def draw_bounding_boxes(im_data: ImageData, ini_bbs_all, bbs_all, ini_ious, ious):
        colors = im_data.VISUALIZATION_COLORS

        im_empty = ((im_data.im_resized / im_data.im_resized.max()) * 255).astype(np.uint8)
        im_empty = np.stack([im_empty] * 3, axis=2)

        im_ini_bbs, im_bbs = {}, {}
        for i_class, classname in enumerate(im_data.CLASSES):
            im_ini_bbs[classname] = im_empty.copy()
            im_bbs[classname] = im_empty.copy()
            ini_bbs, bbs = ini_bbs_all[classname], bbs_all[classname]

            # Number of clusters in labels, ignoring noise if present.
            bb_gts = im_data.get_ground_truths_bbs()[classname]

            # Draw all ground truths bounding boxes
            for im_gt, bb_gt in bb_gts:
                x, y, w, h = bb_gt
                im_ini_bbs[classname] = cv2.rectangle(im_ini_bbs[classname], (x, y), (x + w, y + h), (255, 0, 0), 1)
                im_bbs[classname] = cv2.rectangle(im_bbs[classname], (x, y), (x + w, y + h), (255, 0, 0), 1)

            for k in range(len(bbs)):
                bb_pred = bbs[k]
                bb_ini = ini_bbs[k]

                # identify cluster ground truth defined as the nearest mask
                min_dist_ini = np.max(im_data.im_resized.shape) * 2
                min_dist_pred = np.max(im_data.im_resized.shape) * 2
                nearest_bb_ini = None
                nearest_bb_pred = None

                for im_gt, bb_gt in bb_gts:
                    dist_ini = distance_boundig_boxes(bb_ini, bb_gt)
                    dist_pred = distance_boundig_boxes(bb_pred, bb_gt)
                    if min_dist_ini > dist_ini:
                        min_dist_ini = dist_ini
                        nearest_bb_ini = bb_gt
                    if min_dist_pred > dist_pred:
                        min_dist_pred = dist_pred
                        nearest_bb_pred = bb_gt

                # si no hi ha ground truth
                if nearest_bb_pred is None:
                    iou, iou_ini = 0., 0.

                else:
                    iou = ovni.metrics.IoU_bounding_box(bb_pred, nearest_bb_pred)
                    iou_ini = ovni.metrics.IoU_bounding_box(bb_ini, nearest_bb_ini)

                if ious is not None:
                    ious[classname].append(iou)
                    ini_ious[classname].append(iou_ini)

                x, y, w, h = bb_pred
                im_bbs[classname] = cv2.rectangle(im_bbs[classname], (x, y), (x + w, y + h), colors[i_class], 2)
                cv2.putText(im_bbs[classname], "IoU: {0:.3f}".format(iou), (x, y),
                            cv2.FONT_HERSHEY_PLAIN, 1, color=colors[i_class], thickness=2)

                x, y, w, h = bb_ini
                im_ini_bbs[classname] = cv2.rectangle(im_ini_bbs[classname], (x, y), (x + w, y + h), colors[i_class], 2)
                cv2.putText(im_ini_bbs[classname], "IoU: {0:.3f}".format(iou_ini), (x, y),
                            cv2.FONT_HERSHEY_PLAIN, 1, color=colors[i_class], thickness=2)

        return im_ini_bbs, im_bbs

    def __init__(self, path_run, class_datagen):
        self.class_datagen = class_datagen
        self.class_image = class_datagen.CLASS_IMAGE

        # Root path of the experiments
        self.path_run = path_run

        self.path_data = self.path_run + "data/"
        ovni.check_dir(self.path_data)

        self.path_metrics = self.path_run + "metrics/"
        ovni.check_dir(self.path_metrics)

        self.path_view_stack = self.path_run + 'stack/'
        ovni.check_dir(self.path_view_stack)

        self.path_view_images = self.path_run + 'images/'
        ovni.check_dir(self.path_view_images)

        self.path_tiff_probas = self.path_run + 'tiffs/'
        ovni.check_dir(self.path_tiff_probas)

        self.path_view_proba = self.path_run + "probas/"
        ovni.check_dir(self.path_view_proba)

        self.path_view_hardsamples = self.path_run + "hardsamples/"
        ovni.check_dir(self.path_view_hardsamples)

        self.path_view_contours = self.path_run + "contours/"
        ovni.check_dir(self.path_view_contours)

        self.ephs = None
        self.means = None
        self.covs = None

        self.classifiers = None

    def fit_parameters(self, list_val) -> dict:
        """
        Select the parameters values which maximizes the validation predictions

        :param list_val: List of validation images paths
        :return: Dictionary with the best parameters for each class
        """
        Js = {cl: ([], []) for cl in self.class_image.CLASSES}

        feat_extractors = {cl: MaskFeatures(self.ephs[cl], self.means[cl], self.covs[cl])
                           for cl in self.class_image.CLASSES}
        # iterate over dataset image to get its metrics
        for i, path_im_x in enumerate(list_val):
            image = self.class_image(path_im_x)

            logging.info("Predicting image: " + image.im_name)
            for i_class, classname in enumerate(self.class_image.CLASSES):
                im_gt = image.ground_truth_resized[classname]
                if im_gt is not None:  # ignore images without ground truth
                    im_proba = image.contruct_proba_img(self.classifiers[i_class].clf,
                                                        feat_extractors[classname],
                                                        classname)
                    y_true = im_gt.flatten() != 0
                    y_pred = im_proba.flatten()
                    Js[classname][0].extend(y_true)
                    Js[classname][1].extend(y_pred)

        # identify threshold which maximize the target metric
        thresholds = {}
        for i_class, classname in enumerate(self.class_image.CLASSES):
            y_true, y_pred = Js[classname]

            logging.info("Best threshold for class %s.", classname)
            jacards = []
            ths = np.arange(0.1, 0.9, 0.05)
            for th in ths:
                J = helpers.metrics.jaccard_score(y_true, y_pred > th)
                jacards.append(J)
                logging.info("TH=%f J=%f.", th, J)

            bestThIndex = np.argmax(jacards)
            logging.info("Best config is: TH=%f J=%f.", ths[bestThIndex], jacards[bestThIndex])
            thresholds[classname] = (ths[bestThIndex], jacards[bestThIndex])

        return thresholds

    def train_offsets(self, train_images: list):
        """
        Learn the offsets parametrization according to the train set images

        :param train_images: list of images paths
        :return: None
        """
        # Build a dataframe with the data of all the bounding box in the train set
        data = {
            'x': [], 'y': [], 'w': [], 'h': [], 'class': []
        }

        for i, path_im_x in enumerate(train_images):
            image = self.class_image(path_im_x)
            gts = image.ground_truth_items_resized
            for i_class, classname in enumerate(self.class_image.CLASSES):
                ims_gt = gts[classname]
                for im_gt in ims_gt:
                    if im_gt is not None:
                        x, y, w, h = ovni.masks.bounding_box(im_gt)
                        if x is None:
                            continue

                        data['x'].append(x)
                        data['y'].append(y)
                        data['w'].append(w)
                        data['h'].append(h)
                        data['class'].append(i_class)

        data = pd.DataFrame(data)

        # Compute the interesting metrics from the dataframe
        self.ephs = {}
        self.means = {}
        self.covs = {}
        for i_class, classname in enumerate(self.class_image.CLASSES):
            self.ephs[classname] = (data.loc[data['class'] == i_class, 'h'].max(),
                                    data.loc[data['class'] == i_class, 'w'].max())
            self.means[classname] = (data.loc[data['class'] == i_class, 'h'].mean(),
                                     data.loc[data['class'] == i_class, 'w'].mean())
            self.covs[classname] = np.cov(data.loc[data['class'] == i_class, 'h'],
                                          data.loc[data['class'] == i_class, 'w'])

    def build_classifiers(self, list_train, list_val, rdf_params: dict = None, max_samples_image: int = 1000,
                          n_samples_file: int = int(8e5), load_classifiers: bool = False, hm_iterations: int = 8):
        """
        Build a classifier for each class in the dataset using images in list_train.

        :param list_train: List of train images paths
        :param list_val: List of validation images paths
        :param rdf_params: Classifier parameters
        :param max_samples_image:
        :param n_samples_file:
        :param load_classifiers:
        :param hm_iterations: Number of hard mining iterations
        :return:
        """
        if rdf_params is None:
            rdf_params = {
                'max_depth': config.rf_max_depth,
                'n_jobs': -1
            }
        self.classifiers = []
        metrics = {}

        # Build each one-class classifier
        for i_class, classname in enumerate(self.class_image.CLASSES):

            # Load if exists the last hm iteration classifier
            clf_file = self.path_run + 'clf_' + classname + '_hn_' + str(hm_iterations - 1) + '.sav'
            if os.path.isfile(clf_file) and load_classifiers:
                clf = helpers.ClfWrap(clf_file, rdf_params,
                                      force_clf_creation=False, labels=["Other", classname])

                self.classifiers.append(clf)
                continue

            logging.info("Processing classifier for: %s", classname)

            if (os.path.isfile(self.path_run + "clf_" + classname + ".sav")
                    and load_classifiers and hm_iterations == 0):
                clfwrap = helpers.ClfWrap(self.path_run + "clf_" + classname + ".sav", rdf_params,
                                          force_clf_creation=not load_classifiers, labels=["Other", classname])

            else:

                # If the classifier is not load from disk build it
                gen_data = self.class_datagen(self.path_data, classname, eph=self.ephs[classname],
                                              mean=self.means[classname], cov=self.covs[classname],
                                              max_samples_mask=max_samples_image, n_samples_file=n_samples_file)
                # Generate train and validation data
                gen_data.data_generation(list_train, list_val)
                train_files = gen_data.trainSet.files()
                val_file = gen_data.testSet.files()

                clfwrap, metrs = self.train_and_report(train_files, val_file, rdf_params,
                                                       classname,
                                                       force_clf_creation=True,
                                                       file_clf=self.path_run + "clf_" + classname + ".sav",
                                                       file_report=classname)

                # store metrics of the resulting class classifier
                metrics[classname] = metrs

                if self.class_image.ADD_HN[i_class]:
                    logging.info("Running hard mining of: %s", classname)

                    # init metrics with the initial classifier
                    hn_metrics = []
                    metrs['Iteration'] = 0
                    metrs['n. HN added (% of negatives)'] = 0
                    hn_metrics.append(metrs)

                    # Run hard mining procedure
                    clfwrap, hn_metrics = self.hard_mining(clfwrap, hn_metrics, gen_data, classname, list_train,
                                                           train_files, val_file,
                                                           rdf_params=rdf_params,
                                                           n_iterations=hm_iterations)

                    # Save hard mining results
                    self.write_metrics(hn_metrics,
                                       self.path_metrics + "metrics_hn_" + classname + ".xlsx",
                                       "hard negatives iterations")

            self.classifiers.append(clfwrap)

        return metrics

    def hard_mining(self, clf_hn, hn_metrics: list, gen_data, classname: str, list_train: list,
                    train_files: list, val_files: list, rdf_params: dict,
                    n_iterations=8):
        """
        Perform the hard mining stage of the procedure.

        :param clf_hn: Initial classifier
        :param hn_metrics: Dictionary to fill with the metrics of each hard mining iteration
        :param gen_data: DataGenerator object used to generate the hard negatives data file
        :param classname: Name of the target class
        :param list_train: list of training images files path
        :param train_files: list of training data files path
        :param val_files: list of validation data files path
        :param rdf_params: Dictionary with the RDF parametrization
        :param n_iterations: Number of hard mining iterations to perform
        :return: builded classifier and dictionary with iterations metrics
        """
        hn_mining = [0.2] * n_iterations
        for i, perc_hn in enumerate(hn_mining):
            gen_data.hardNegativesGeneration(clf_hn.clf, list_train, perc_hn=perc_hn)

            # gen_data.hardNegativesInclusion(clf_hn, perc_hn=perc_hn)
            gen_data.HardNegativesAppend(perc_hn=perc_hn)

            clf_hn, metrs = self.train_and_report(train_files, val_files, rdf_params,
                                                  classname,
                                                  self.path_run + 'clf_' + classname + '_hn_' + str(i) + '.sav',
                                                  classname + "_hn_" + str(i),
                                                  force_clf_creation=True)

            metrs['Iteration'] = i + 1
            metrs['n. HN added (% of negatives)'] = perc_hn
            metrs['P'] = clf_hn.train_P
            metrs['N'] = clf_hn.train_N

            hn_metrics.append(metrs)

        return clf_hn, hn_metrics

    def rdf_params_evaluation(self, list_train, list_val, test_params: dict, rdf_params=None,
                              max_samples_image: int = 1000, n_samples_file: int = int(8e5)):
        def run_config(v, rdf_parameters, classname, train_files, val_files, reload_data):
            ini_time = timer()
            if reload_data:
                gen_data = self.class_datagen(self.path_data, classname, eph=self.ephs[classname],
                                                          mean=self.means[classname], cov=self.covs[classname],
                                                          max_samples_mask=config.DATA_MAX_PIXELS_CLASS,
                                                          n_samples_file=config.MAX_SAMPLES_FILE)

                gen_data.data_generation(list_train, list_val,
                                         n_train_files=config.N_TRAIN_FILES,
                                         n_test_files=config.N_TEST_FILES,
                                         max_train_files=20)
                train_files = gen_data.trainSet.files()
                val_files = gen_data.testSet.files()
            end_time_data_build = timer()

            clf_hn, metrs = self.train_and_report(train_files, val_files, rdf_parameters,
                                                  classname,
                                                  self.path_run + "clf_" + classname + ".sav",
                                                  file_report=classname,
                                                  force_clf_creation=True)
            metrs['value'] = v
            metrs['time data build (s)'] = end_time_data_build - ini_time
            return metrs

        if rdf_params is None:
            rdf_params = {
                'max_depth': config.rf_max_depth,
                'n_jobs': -1
            }

        for i_class, classname in enumerate(self.class_image.CLASSES):
            metrics_file = self.path_metrics + "metrics_params_" + classname + ".xlsx"
            gen_data = self.class_datagen(self.path_data, classname, eph=self.ephs[classname],
                                                      mean=self.means[classname], cov=self.covs[classname],
                                                      max_samples_mask=max_samples_image,
                                                      n_samples_file=n_samples_file)

            gen_data.data_generation(list_train, list_val, max_train_files=20)

            train_files = gen_data.trainSet.files()
            test_file = gen_data.testSet.files()

            for param, values in test_params['rdf'].items():
                default_value = rdf_params[param] if param in rdf_params else None
                metrics = []
                for v in values:
                    rdf_params[param] = v

                    metrs = run_config(v, rdf_params, classname, train_files, test_file, False)
                    metrics.append(metrs)

                self.write_metrics(metrics, metrics_file, sheet_name=param)

                if default_value is None:
                    rdf_params.pop(param)
                else:
                    rdf_params[param] = default_value

            for param, values in test_params['procedure'].items():
                default_value = getattr(config, param)
                rdf_params_proc = rdf_params.copy()
                metrics = []
                for v in values['values']:
                    setattr(config, param, v)
                    logging.info("testing %s with %s", param, str(v))
                    if param == 'rf_inc_trees_fit':
                        rdf_params_proc['n_jobs'] = 1
                        print(config.rf_inc_trees_fit)
                        print(rdf_params_proc)

                    metrs = run_config(v, rdf_params_proc, classname, train_files, test_file, values['reload_data'])
                    metrics.append(metrs)

                self.write_metrics(metrics, metrics_file, sheet_name=param)

                setattr(config, param, default_value)

    def visualization(self, list_imgs, ths, bb_sizes):

        predictions = {cl: ([], []) for cl in self.class_image.CLASSES}

        feat_extractors = {cl: MaskFeatures(self.ephs[cl], self.means[cl], self.covs[cl])
                           for cl in self.class_image.CLASSES}
        ious = {cl: [] for cl in self.class_image.CLASSES}
        ini_ious = {cl: [] for cl in self.class_image.CLASSES}
        ious_segm_acum = {cl: 0.0 for cl in self.class_image.CLASSES}

        precision_acum = {cl: 0.0 for cl in self.class_image.CLASSES}
        recall_acum = {cl: 0.0 for cl in self.class_image.CLASSES}
        image_counter = {cl: 0 for cl in self.class_image.CLASSES}

        timings = {
            'predict': [],
            'process': [],
            'clustering': [],
            'bbs': [],
            'total': [],
            'class': []
        }
        for i, path_im_x in enumerate(list_imgs):
            image = self.class_image(path_im_x)

            # Process image
            ini_bbs_all, bbs_all = image.process_image(self.classifiers, feat_extractors, ths, bb_sizes, timings)
            for i_class, classname in enumerate(self.class_image.CLASSES):
                ini_bbs_all[classname] = non_max_suppression_fast(np.asarray(ini_bbs_all[classname]), 0.1)
                bbs_all[classname] = non_max_suppression_fast(np.asarray(bbs_all[classname]), 0.1)

            # Draw bounding boxes
            im_ini_bbs, im_bbs = self.draw_bounding_boxes(image, ini_bbs_all, bbs_all, ini_ious, ious)

            # Generating class boundaries images
            im_contours = ((image.im_resized - image.im_resized.min()) / (image.im_resized - image.im_resized.min()).max()) * 255
            im_contours = cv2.cvtColor(im_contours.astype(np.uint8), cv2.COLOR_GRAY2RGB)
            for i_class, classname in enumerate(self.class_image.CLASSES):
                tifffile.imwrite(self.path_tiff_probas + classname + "_" + image.im_name + ".tif",
                                 data=image.proba_images[classname])
                if image.ground_truth_resized[classname] is not None:
                    ovni.draw_mask_boundaries(im_contours, image.ground_truth_resized[classname], (0, 0, 0))
                ovni.draw_mask_boundaries(im_contours, image.masks_prob[classname],
                                          self.class_image.VISUALIZATION_COLORS[i_class])

            # Computing segmentation IoU
            for i_class, classname in enumerate(self.class_image.CLASSES):
                im_gt = image.ground_truth_resized[classname]
                if im_gt is not None:
                    y_true = (im_gt != 0).flatten()
                    # If not positives values in ground truth metrics return 0
                    # To avoid this bias we ignore those images
                    if np.count_nonzero(y_true) > 0:
                        y_pred = image.masks_prob[classname].flatten()
                        J = helpers.metrics.jaccard_score(y_true, y_pred)
                        prec = helpers.metrics.precision_score(y_true, y_pred)
                        recall = helpers.metrics.recall_score(y_true, y_pred)

                        ious_segm_acum[classname] += J
                        precision_acum[classname] += prec
                        recall_acum[classname] += recall
                        image_counter[classname] += 1

            # Hard negatives drawn
            for i_class, classname in enumerate(self.class_image.CLASSES):
                im_gt = image.ground_truth_resized[classname]
                if im_gt is None:
                    continue

                y_true = (im_gt != 0)
                im_proba = image.proba_images[classname]

                hn = (y_true == 0) & (im_proba > 0.8)
                positions = np.nonzero(hn)

                im_hn = ((image.im_resized / image.im_resized.max()) * 255).astype(np.uint8)
                im_hn = np.stack([im_hn] * 3, axis=2)
                im_hn[positions] = (0, 0, 255)
                plt.imsave(self.path_view_hardsamples + image.im_name + "_" + classname + ".png",
                           im_hn, cmap="Greys")

            plt.imsave(self.path_view_contours + image.im_name + ".png", im_contours, cmap="Greys")
            images = [
                {
                    'n_row': 0,
                    "img": image.im_resized,
                    'cmap': 'gray',
                    "title": "Input"
                }
            ]
            for i_class, classname in enumerate(self.class_image.CLASSES):
                im_proba = image.proba_images[classname]
                im_proba[1, 1] = 1.0
                images.append({
                    'n_row': 0,
                    "img": image.ground_truth[classname],
                    "title": "Ground truth",
                    "classname": classname,
                    "cmap": "jet"
                })
                images.append({
                    'n_row': i_class + 1,
                    "img": im_proba,
                    "title": "Probab",
                    "classname": classname,
                    "cmap": "jet"
                })
                images.append({
                    'n_row': i_class + 1,
                    "img": image.masks_prob[classname],
                    "title": "Binary",
                    "classname": classname,
                    'cmap': 'gray',
                })
                images.append({
                    'n_row': i_class + 1,
                    "img": im_contours,
                    "title": "ContoursThresh",
                    "classname": classname,
                    "cmap": "jet"
                })
                images.append({
                    'n_row': i_class + 1,
                    "img": image.build_cluster_image(image.clusters[classname],
                                                       image.clusters_data[classname]),
                    "title": "Clusters",
                    "classname": classname,
                    "cmap": "jet"
                })
                images.append({
                    'n_row': i_class + 1,
                    "img": im_ini_bbs[classname],
                    "title": "Initial bbs",
                    "classname": classname,
                    "cmap": "gray"
                })
                images.append({
                    'n_row': i_class + 1,
                    "img": im_bbs[classname],
                    "title": "BoundingBoxes",
                    "classname": classname,
                    "cmap": "gray"
                })
                im_gt = image.ground_truth_resized[classname]
                if im_gt is not None:
                    y_true = im_gt.flatten() != 0
                    y_pred = im_proba.flatten()
                    J = helpers.metrics.jaccard_score(y_true, y_pred > ths[classname][0])
                    predictions[classname][0].extend(y_true)
                    predictions[classname][1].extend(y_pred)
                    logging.info(classname + " J = " + str(J))

            n_imgs_cols = 6
            n_imgs_rows = len(self.class_image.CLASSES) + 1
            plotter = ovni.MultiPlot(n_img=(n_imgs_rows, n_imgs_cols))
            plotter.save_multiplot(self.path_view_proba + image.im_name, images)
            plt.close()

            for im_view in images:
                im = im_view['img']
                fig = plt.figure(frameon=False)
                max_ax = np.max(im.shape[:2])
                fig.set_size_inches((im.shape[1] / max_ax) * 4, (im.shape[0] / max_ax) * 4)
                ax = plt.Axes(fig, [0., 0., 1., 1.])
                ax.set_axis_off()
                fig.add_axes(ax)
                cmap = im_view['cmap'] if 'cmap' in im_view else None
                ax.imshow(im, cmap=cmap, aspect='auto')
                fname = self.path_view_images + image.im_name + "_"
                if "classname" in im_view:
                    fname += im_view['classname'] + '_' + im_view['title'] + ".png"

                else:
                    fname += im_view['title'] + ".png"

                fig.savefig(fname)
                plt.close()

        df_timings = pd.DataFrame(timings)
        metrics = []
        for i_class, classname in enumerate(self.class_image.CLASSES):
            y_true, y_pred = predictions[classname]
            y_pred = np.array(y_pred) > ths[classname][0]
            J = helpers.metrics.jaccard_score(y_true, y_pred)
            self.classifiers[i_class].plot_report(y_true, y_pred, report_filename=classname + "_testset")
            cm = helpers.metrics.confusion_matrix(y_true, y_pred)
            np.save(self.path_metrics + "/cm_" + classname + "_testset.npy", cm)

            precision, recall, f1, support = helpers.metrics.precision_recall_fscore_support(y_true, y_pred)
            prec_avg, rec_avg, f1_avg, _ = helpers.metrics.precision_recall_fscore_support(y_true, y_pred,
                                                                                           average="weighted")
            cohen_kappa = helpers.metrics.cohen_kappa_score(y_true, y_pred)
            metrics.append({
                'Classifier': classname,
                'Jaccard': J,
                'cohen_kappa': cohen_kappa,
                'Precision other': precision[0],
                'Recall other': recall[0],
                'F1 other': f1[0],
                'Support other': support[0],
                'Precision target': precision[1],
                'Recall target': recall[1],
                'F1 target': f1[1],
                'Support target': support[1],
                'Precision Weighted average': prec_avg,
                'Recall Weighted average': rec_avg,
                'F1 Weighted average': f1_avg,
                'IoU bbs': np.mean(np.array(ious[classname])),
                'IoU ini bbs': np.mean(np.array(ini_ious[classname])),
                'mean IoU segmentation': ious_segm_acum[classname] / image_counter[classname],
                'mean Precision segmentation': precision_acum[classname] / image_counter[classname],
                'mean Recall segmentation': recall_acum[classname] / image_counter[classname],
                'mean processing time': df_timings.loc[df_timings['class'] == classname, 'total'].mean()
            })

        self.write_metrics(metrics, self.path_metrics + "metrics_testset.xlsx", "metrics")
        self.write_metrics(timings, self.path_metrics + "metrics_testset.xlsx", "timings")

        return metrics
