#!/usr/bin/env python

from distutils.core import setup

try:
    from catkin_pkg.python_setup import generate_distutils_setup
    setup_args = generate_distutils_setup(
        packages=['promp'],
        package_dir={'': 'src'})
    setup(**setup_args)

except ImportError:
    setup(name='Probabilistic Movement Primitives',
          version='0.1.0',
          description='Library to create and use probabilistic motion primitives, which are distributions over continous trajectories',
          author='Raphael Deimel',
          author_email='raphael.deimel@tu-berlin.de',
          url='http://www.mti-engage.tu-berlin.de/',
          packages=['promp'],
          package_dir={'': 'src'},
          install_requires=['numpy', 'scipy', 'sklearn', 'hdf5storage']
         )

