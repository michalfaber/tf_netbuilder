from setuptools import setup, find_packages

exec(open('tf_netbuilder/version.py').read())
setup(
   name='tf_netbuilder',
   version='1.0',
   description='Builder of neural net architectures from text for Tensorflow 2.0',
   author='Michal Faber',
   author_email='michal@worldestimators.com',
   url='https://github.com/michalfaber/tf_netbuilder',
   packages=find_packages(exclude=['examples', 'resources']),
   install_requires=['tensorflow >= 2.0.0'],
   python_requires='>=3.6',
   classifiers=[
      # How mature is this project? Common values are
      #   3 - Alpha
      #   4 - Beta
      #   5 - Production/Stable
      'Development Status :: 3 - Alpha',
      'Intended Audience :: Education',
      'Intended Audience :: Science/Research',
      'License :: OSI Approved :: Apache Software License',
      'Programming Language :: Python :: 3.6',
      'Programming Language :: Python :: 3.7',
      'Programming Language :: Python :: 3.8',
      'Topic :: Scientific/Engineering',
      'Topic :: Scientific/Engineering :: Artificial Intelligence',
      'Topic :: Software Development',
      'Topic :: Software Development :: Libraries',
      'Topic :: Software Development :: Libraries :: Python Modules',
   ],

   keywords='tensorflow netbuilder openpose mobilenetv3',
)