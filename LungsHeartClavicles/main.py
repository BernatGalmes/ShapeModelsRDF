import logging
from glob import glob
import pandas as pd

from LungsHeartClavicles.LHCDataGenerator import LHCDataGenerator
from LungsHeartClavicles.LHCImage import LHCImage

from shapemodels.Config import config
from shapemodels.Procedure import Procedure
from shapemodels.Image import ImageData

# set the folder path where the results will be stored
config.FOLDER_DATA = '/media/bernat/Data/rdf-segmentation/data'

# set the folder path of the dataset images.
# The target folder must have two subfolders called 'All247images' and 'masks'
# with the input and the ground truth images respectively
LHCImage.PATH_DATASET = '/home/bernat/datasets/LungsHeartClavicles/'

# Set it to true to load the classifiers from the previous training
LOAD_CLASSIFIERS = False

# name of the results folder used to identify the results folder
EXPERIMENT_NAME = "experiment_name"

# Path to store the results
path_results = config.FOLDER_RESULTS + "LHC/" + EXPERIMENT_NAME + "/"


def main():
    log = logging.getLogger(__name__)
    logging.basicConfig(level=logging.DEBUG)
    logging.getLogger('matplotlib').setLevel(logging.WARNING)

    """
    Les imatges de raigs x mostren la densitat dels distints elements que
    apareixen en les imatges
    """
    # save_description(path_results)

    print(LHCImage.PATH_DATASET_INPUT + '/*.IMG')
    list_imgs_folders = glob(LHCImage.PATH_DATASET_INPUT + '/*.IMG')

    logging.info("Normal run ... ")
    list_train = list_imgs_folders[:50]
    list_val = list_imgs_folders[150:160]
    list_test = list_imgs_folders[200:210]

    proc = Procedure(path_results, LHCDataGenerator)
    proc.train_offsets(list_train)
    ephs, means, covs = proc.ephs, proc.means, proc.covs

    logging.debug("n train: %i, n val: %i, n. test: %i", len(list_train), len(list_val), len(list_test))

    rdf_params = {
        'max_depth': config.rf_max_depth,
        'max_features': 0.2,
        'n_jobs': -1
    }
    metrics = proc.build_classifiers(list_train, list_val, rdf_params=rdf_params, load_classifiers=LOAD_CLASSIFIERS,
                                     max_samples_image=100, n_samples_file=int(4e5), hm_iterations=1)

    logging.info("Adjusting parameters ...")
    ths = proc.fit_parameters(list_val)
    print(ths)

    ImageData.set_opening_kernel(None)
    ImageData.set_clossing_kernel(None)
    metrics = proc.visualization(list_test, ths, means)

    metrics_df = pd.DataFrame(metrics)
    with pd.ExcelWriter(proc.path_metrics + "metrics_testset_postproc.xlsx") as writer:
        metrics_df.to_excel(writer, sheet_name='NoMorph')


main()
