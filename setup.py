from setuptools import setup
import sys

assert sys.version_info.major == 3 and sys.version_info.minor >= 6, \
    "The cyc_near_miss repo is designed to work with Python 3.6 and greater." \
    + "Please install it before proceeding."

setup(
    name='Cycling Near-Miss',
    py_modules=['cnm'],
    version='0.1',
    install_requires=[
        'progressbar',
        'tqdm',
        'pandas',
        'numpy',
        'scikit-learn'
    ],
    description="A Benchmark for Cycling Near Miss Detection from Video Streams.",
    author="Lingheng Meng, Mingjie Li, Zijue Chen",
)
