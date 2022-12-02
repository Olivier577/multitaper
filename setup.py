# Copyright 2022 Germán A. Prieto, MIT license

from setuptools import setup, find_packages
from Cython.Build import cythonize # added by om

setup(
    name="multitaper",
    version="1.1.5",
    author="German A. Prieto ",
    author_email="gaprietogo@unal.edu.co",
    description="Multitaper codes translated into Python",
    long_description_content_type="text/markdown",
    url="https://github.com/gaprieto/multitaper",
    classifiers=[
        "Programming Language :: Python :: 3.8",
        "Operating System :: OS Independent",
    ],
    packages=find_packages(),
    package_data={
        "multitaper": [
            "examples/*.ipynb",
            "examples/*.py",
            "examples/figures/*.jpg",
            "examples/figures/*.png"
        ]
    },
    python_requires=">=3.7",
    ext_modules=cythonize("multitaper/src_om/*pyx"), # added by om
)


#---------------------------------------------
# On a Mac adding this creates a problem
# but should be added for completeness.
#
#     install_requires=['numpy', 'scipy'],
#
#---------------------------------------------
