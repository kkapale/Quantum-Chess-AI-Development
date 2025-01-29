from setuptools import setup
from Cython.Build import cythonize
import numpy as np

setup(
    name='MCTS_Cython',
    ext_modules=cythonize("MCTS_NN.pyx", language_level=3),
    include_dirs=[np.get_include()],
    zip_safe=False,
)
