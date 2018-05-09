from setuptools import setup
from distutils.extension import Extension
import numpy as np


setup(
    name='LB_colony_growth',
    version='0.1',
    packages=['LB_colony_growth'],
    include_package_data=True,
    url='',
    license='',
    author='Bryan Weinstein',
    author_email='btweinstein@gmail.com',
    description='',
    include_dirs = [np.get_include()],
    requires=['pyopencl', 'numpy']
)
