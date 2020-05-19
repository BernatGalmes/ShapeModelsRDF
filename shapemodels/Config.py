import json
import argparse


class Config:
    """
        Class with the default values used in the framework parameters
    """

    __INSTANCE = None

    class __Data:

        def __init__(self):

            # Software version
            self.VERSION = 1

            # Execution parameters
            self.OFFSET_MAX = 1

            self.N_FEATURES = 500
            self.N_OFFSETS = 1

            self.OFFSETS_SEED = 10

            self.features_factor = 1.0
            self.DATA_MAX_PIXELS_CLASS = 200

            self.MAX_SAMPLES_FILE = int(8e5)
            self.N_TRAIN_FILES = None
            self.N_TEST_FILES = None
            self.MAX_TRAIN_FILES = 10

            # Random forest parameters
            self.rf_inc_trees_fit = 10
            self.rf_max_depth = 20
            self.rf_min_samples_leaf = 1
            self.rf_n_jobs = 1
            self.rf_class_weight = 'balanced'

            # File system parameters
            self.__FOLDER_DATA = ''

        @property
        def FOLDER_DATA(self):
            return self.__FOLDER_DATA

        @FOLDER_DATA.setter
        def FOLDER_DATA(self, val):
            self.__FOLDER_DATA = val

        @property
        def DATA_COLUMNS(self):
            """
            Llista de les columnes de les dades a utilitzar de cada regió
            :return:
            """
            return ["feat_" + str(i) for i in range(0, self.N_FEATURES)]

        @property
        def FOLDER_RESULTS(self):
            return self.FOLDER_DATA + "/results/v" + str(self.VERSION) + "/"

        def set_arguments(self, parser=None):
            if parser is None:
                parser = argparse.ArgumentParser()
            parser.add_argument('--n_features', help='foo help')

            args = parser.parse_args()
            if args.n_features:
                config.N_FEATURES = int(args.n_features)

            return args

        def save_json(self, file):
            with open(file, 'w') as fp:
                json.dump(self.__dict__, fp, sort_keys=True, indent=4)

        def save_txt(self, path):
            file = path + "config.info"
            with open(file, 'w') as fp:
                fp.write(self.__str__())

        def __str__(self):
            txt = "----------" \
                  "\nPROGRAM CONFIGURATION:" \
                  "\n----------"
            for key in self.__dict__:
                txt += key + ": " + str(self.__dict__[key]) + "\n"

            return txt

    def __new__(cls, *args, **kwargs):
        if cls.__INSTANCE is None:
            cls.__INSTANCE = cls.__Data()
        return cls.__INSTANCE


config = Config()
