from setuptools import setup, find_packages
import codecs
import os

VERSION = '0.0.10.post1'
DESCRIPTION = 'Linearly referenced data management, manipulation, and operations'
with open('README.rst') as f:
    LONG_DESCRIPTION = ''.join(f.readlines())

# Setting up
setup(
    name="linref",
    version=VERSION,
    author="Tariq Shihadah",
    author_email="<tariq.shihadah@gmail.com>",
    description=DESCRIPTION,
#    long_description_content_type="text/x-rst",
    long_description=LONG_DESCRIPTION,
    packages=find_packages(),
    package_data={"": ["*.json"]},
    install_requires=['numpy', 'matplotlib', 'shapely>=1.7', 'pandas>=1.1', 'geopandas>=0.10.2', 'rangel>=0.0.7', 'sphinx', 'myst_parser'],
    keywords=['python', 'geospatial', 'linear', 'data', 'event', 'dissolve', 'overlay', 'range', 'numeric', 'interval'],
    classifiers=[
        "Development Status :: 1 - Planning",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: Unix",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows",
    ]
)
