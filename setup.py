from setuptools import setup
from distutils.extension import Extension
import numpy as np


setup(
    name='LB_colony_growth',
    version='0.1',
    packages=['LB_colony_growth'],
    include_package_data=True,
    url='https://github.com/btweinstein/LB_colony_growth',
    license='',
    author='Bryan Weinstein',
    author_email='btweinstein@gmail.com',
    description='Growing DLA colonies on a lattice with coupling to a Lattice-Boltzmann based nutrient field.',
    include_dirs = [np.get_include()],
    install_requires=['pyopencl', 'numpy', 'mayavi', 'mako']
)
