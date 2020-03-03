#!/usr/bin/env python

from distutils.core import setup

packages=['promep', 'mechanicalstate', 'staticprimitives', 'namedtensors']
package_dir={'': 'src'}

try:
    from catkin_pkg.python_setup import generate_distutils_setup
    setup_args = generate_distutils_setup(
        packages=packages,
        package_dir=package_dir)
    setup(**setup_args)

except ImportError:
    setup(name='Probabilistic Movement Primitives',
          version='0.1.0',
          description='Library to create and use probabilistic motion primitives, which are distributions over continous trajectories',
          author='Raphael Deimel',
          author_email='raphael.deimel@tu-berlin.de',
          url='http://www.mti-engage.tu-berlin.de/',
          packages=packages,
          package_dir=package_dir,
          install_requires=['numpy', 'scipy', 'sklearn', 'hdf5storage', 'pandas']
         )

