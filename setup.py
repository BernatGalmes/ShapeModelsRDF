from setuptools import setup

setup(name='shapemodels',
      version='0.0.1',
      description='Classification framework to detect and recognize objects in single channel images',
      url='',
      author="Bernat Galmés Rubert, Dr. Gabriel Moyà Alcover",
      author_email='bernat_galmes@hotmail.com, gabriel_moya@uib.es',
      license='MIT',
      packages=['shapemodels'],
      install_requires=[
          'scikit-learn',
          'matplotlib',
          'seaborn',
          'pandas',
          'seaborn',
          'numpy',
          'scikit-image',
          'opencv-python',
          'ovnimage'
      ],
      zip_safe=False)