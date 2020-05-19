# ShapeModelsRDF
Classification framework to detect and recognize objects in single channel images.

# Main functionality

The core functionality is split in several classes with different objectives. 
 
## Constants
The default values used in the procedure configuration are encompassed in the Config.py file.

The class Config contained in that file follow a singleton structure, 
then each object instantiated has the same values as previous instances.

You should modify class attributes to change default config values.

Normally the class is used directly with the object instance included in the file,
as the next example:
```python
from shapemodels.Config import config

config.FOLDER_DATA = '/media/bernat/Data/rdf-segmentation/data'

``` 

## Elements
The next sections detail the main classes contained in the project.

### MaskFeatures 
Class in charge of carrying out de features computation of an image.


### ImageData
Class which represent a dataset image instance.

You should implement in a child class, the following methods, which vary for each dataset:

```python

from shapemodels.Image import ImageData

class LHCImage(ImageData):
    PATH_DATASET = ''

    RESIZE_PARAMS = dict(dsize=(256, 256), fx=0, fy=0)

    CLASSES = []

    def _load_from_path(self, path) -> tuple:
        ...

    @property
    def ground_truth_items(self) -> dict:
        ...

    @property
    def ground_truth(self) -> dict:
        ...

``` 

* PATH_DATASET: Is not required, but it's useful to identify here the path of the dataset.
* REZIZE_PARAMS: Specification of a dictionary with the parameters of cv2.resize function. You should specify here
how do you want to resize the images.
* CLASSES: List of strings. Names of the classes in the dataset.
* _load_from_path: Function to implement. Should read the input image, and its respective labels, from the given path. 
And return a 2-tuple with the read image and its labels, in this order. 
* ground_truth_items: Property to implement. Should compute a dictionary from the class labels, which should contain
for each class a binary image with its ground truth.
* ground_truth: Property to implement. Should compute a dictionary from the class labels, which should contain
for each class a list of binary images each one with the ground truth of each object instance appearing in the image.

### DataGenerator
Class occupied to generate the data to use to train and validate the classifier.

You should implement a child class specifying its child Image class, the next example shows the implementation
from the previous image class example.

```python
from LungsHeartClavicles.LHCImage import LHCImage
from shapemodels.DataGenerator import DataGenerator

class LHCDataGenerator(DataGenerator):

    TH_HARD_SAMPLE = 0.9
    CLASS_IMAGE = LHCImage

``` 

### Procedure
Class containing the procedure algorithm. View use case section to inspect is usage.

### Helpers
Classes and functions used internally by the framework.

# Use case
In the repository you are going to see an example use of the software in folder: LungsHeartClavicles.

The next piece of code show you how to use the classification framework.
```python
from shapemodels.Procedure import Procedure
from LungsHeartClavicles.LHCDataGenerator import LHCDataGenerator

# Path to folder where to store the results
path_results = ''

# List of image path to use to train
list_train = []

# List of image path to use to validate
list_val = []

# List of image path to use to test
list_test = []

# init instance
proc = Procedure(path_results, LHCDataGenerator)

# Learn classifier priors
proc.train_offsets(list_train)

# Train or load classifier
metrics = proc.build_classifiers(list_train, list_val)

# Compute classifiers thresholds maximizing metrics
ths = proc.fit_parameters(list_val)

# Predict the test images to visualize (Results are stored in 'path_results')
metrics = proc.visualization(list_test, ths, proc.means)

``` 

# Sample dataset

[link to download](https://www.dropbox.com/s/svux2wjfi71tw83/LungsHeartClavicles.zip?dl=1)
