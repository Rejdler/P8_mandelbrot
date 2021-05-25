# -*- coding: utf-8 -*-
"""
Created on Sun May  2 23:55:14 2021

@author: dksan
"""

from setuptools import setup
from Cython.Build import cythonize

setup(
    name ='Mandel',
    ext_modules = cythonize("Mandelbrot_naive_cython.py"),
    zip_safe = False
    #ext_modules = cythonize("Hello.py")
)