from os.path import dirname, exists, realpath
from setuptools import setup, find_packages
import sys

author = "Brian Ashcroft"
authors = [author]
description = 'A set of scripts to analyze, flatten and load AFM files from a variety of instruments'
name = 'pyFlammarion'
year = "2025"


sys.path.insert(0, realpath(dirname(__file__))+"/"+name)
try:
    from _version import version   
except BaseException:
    version = "unknown"

#!pip install afmformats
#!pip install matplotlib
#!pip install numpy
#!pip install scipy
#!pip install scikit-image
#!pip install pandas
setup(
    name=name,
    author=author,
    author_email='brian.ashcroft@asu.edu',
    url='https://github.com/RecognitionAnalytics/pyFlammarion',
    version=version,
    packages=find_packages(),
    package_dir={name: name},
    include_package_data=True,
    license="GPLv3",
    description=description,
    long_description=open('README.rst').read() if exists('README.rst') else '',
    install_requires=["h5py",
                      "igor2>=0.5.0",  # Asylum Research .ibw file format
                      "jprops",  # JPK file format
                      "numpy>=1.14.0",
                      "matplotlib",
                      "scipy",
                      "scikit-image",
                      "pandas",
                      "scikit-learn",
                      ],
    python_requires='>=3.6, <4',
    keywords=["atomic force microscopy",
              "flattening",
              "break junction",
              "mask generation",
              "molecular imaging",
              "force spectroscopy",
              "agilent",
              ],
    classifiers=[
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3',
        'Topic :: Scientific/Engineering :: Visualization',
        'Intended Audience :: Science/Research'
    ],
    platforms=['ALL'],
)