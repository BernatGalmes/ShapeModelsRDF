import os
import pickle
import logging

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
import sklearn.metrics as metrics

from shapemodels.Config import config

import seaborn as sn
import matplotlib.pyplot as plt

import ovnImage as ovni


def save_description(path_results):
    description = input("Describe el experimento que vas a llevar a cabo:")
    description += "\n" + str(config)
    with open(path_results + "/readme.txt", "w") as text_file:
        text_file.write(description)


class ClfWrap:
    def __init__(self, file_clf, rdf_params=None, labels=None, force_clf_creation=False):
        if rdf_params is None:
            self.rdf_params = {}

        else:
            self.rdf_params = rdf_params

        # Set RDF default params
        self.rdf_params['n_estimators'] = 0
        self.rdf_params['warm_start'] = True

        if 'max_depth' not in self.rdf_params:
            self.rdf_params['max_depth'] = config.rf_max_depth

        if 'min_samples_leaf' not in self.rdf_params:
            self.rdf_params['min_samples_leaf'] = config.rf_min_samples_leaf

        if 'n_jobs' not in self.rdf_params:
            self.rdf_params['n_jobs'] = config.rf_n_jobs

        if 'class_weight' not in self.rdf_params:
            self.rdf_params['class_weight'] = config.rf_class_weight

        if 'criterion' not in self.rdf_params:
            self.rdf_params['criterion'] = 'entropy'

        if 'verbose' not in self.rdf_params:
            self.rdf_params['verbose'] = 0

        if 'random_state' not in self.rdf_params:
            self.rdf_params['random_state'] = 10

        self.y_true = None
        self.y_pred = None
        self.labels = labels

        self.train_N = 0
        self.train_P = 0
        self.folder_clf = os.path.dirname(file_clf)
        self.folder_metrics = self.folder_clf + os.path.sep + "metrics" + os.path.sep
        ovni.check_dir(self.folder_metrics)

        self.path_clf = file_clf
        if os.path.isfile(file_clf) and not force_clf_creation:
            with open(file_clf, 'rb') as handle:
                self.clf = pickle.load(handle)
            self.loaded = True

            logging.info("Classifier loaded from file")
        else:
            self.loaded = False
            self.clf = RandomForestClassifier(**self.rdf_params)

    def generate(self, train_files, test_file, trees4file=config.rf_inc_trees_fit,
                 do_prediction=True):

        if self.clf.n_estimators > 0:
            return self.clf

        self.train(train_files, trees4file)

        if do_prediction:
            y_pred, y_real = self.predict(test_file)

            self.plot_report(y_real, y_pred,
                             os.path.basename(self.path_clf).split('.')[0])

        return self.clf

    def plot_report(self, y_real, y_pred, report_filename=None):
        stats = ovni.metrics.print_classification_stats(y_real, y_pred, self.labels)

        if report_filename is not None:
            clf_report = metrics.classification_report(y_real, y_pred, target_names=self.labels)
            with open(self.folder_metrics + "/metrics_" + report_filename + ".txt", "w") as text_file:
                text_file.write(clf_report)
            plt.clf()
            sn.set(font_scale=1.4)  # for label size

            confusion_matrix = metrics.confusion_matrix(y_real, y_pred)
            np.save(self.folder_metrics + "/cm_" + report_filename + ".npy", confusion_matrix)
            confusion_matrix = pd.DataFrame(confusion_matrix)
            if self.labels is not None:
                confusion_matrix.index = self.labels
                confusion_matrix.columns = self.labels
            sn.heatmap(confusion_matrix, annot=True, annot_kws={"size": 9})  # font size
            plt.savefig(self.folder_metrics + "/cm_" + report_filename + ".png")
        return stats

    def train(self, train_files, trees4file=config.rf_inc_trees_fit):
        self.train_N = 0
        self.train_P = 0
        for train_file in train_files:
            train_data = np.load(train_file)
            y_train = train_data[:, -1]

            logging.info("training data ... " + train_file)
            N = np.count_nonzero(y_train == 0)
            P = np.count_nonzero(y_train != 0)
            self.train_N += N
            self.train_P += P
            logging.info("N. positives: %d -- N. negatives: %d", P, N)

            self.clf.n_estimators += trees4file
            self.clf.fit(train_data[:, :-1], y_train)

        with open(self.path_clf, 'wb') as handle:
            pickle.dump(self.clf, handle)

    def getHardNegativeSamples(self, test_file):
        ...

    def predict(self, test_file):
        logging.info("predicting data ...")
        if not isinstance(test_file, list):
            test_data = np.load(test_file)
            y_real = test_data[:, -1]
            y_pred = self.clf.predict(test_data[:, :-1])

        else:
            y_pred, y_real = [], []
            for file in test_file:
                test_data = np.load(file)
                y = test_data[:, -1]
                y_pred.extend(
                    list(self.clf.predict(test_data[:, :-1]))
                )
                y_real.extend(list(y))

        return y_real, y_pred


def fig2data(fig):
    """
    @brief Convert a Matplotlib figure to a 4D numpy array with RGBA channels and return it
    @param fig a matplotlib figure
    @return a numpy 3D array of RGBA values
    """
    # If we haven't already shown or saved the plot, then we need to
    # draw the figure first...
    fig.canvas.draw()

    # Now we can save it to a numpy array.
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    return data
