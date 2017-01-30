from setuptools import setup
from setuptools import find_packages, Extension, setup  # noqa: F811

import numpy.distutils.misc_util
import os
from glob import glob

include_dirs = numpy.distutils.misc_util.get_numpy_include_dirs()
sources = glob('seq_lda/_seq_lda/*.cpp')

seq_lda = Extension('seq_lda._seq_lda',
                    define_macros=[('MAJOR_VERSION', '1'),
                                   ('MINOR_VERSION', '0')],
                    include_dirs=include_dirs,
                    libraries=[],
                    sources=sources,
                    extra_compile_args=['--std=c++11 -Wno-sign-compare'],
                    language='c++')

setup(name='seq_lda',
      version='1.0',
      description='Latent Dirichlet Allocation for sequential data.',
      author='Eric Crawford',
      author_email='eric.crawford@mail.mcgill.ca',
      packages=find_packages(),
      include_package_data=True,
      url='',
      long_description='''
A package for using LDA for multitask sequence learning.
LDA implementation based on lda-c by David Blei.
''',
      ext_modules=[seq_lda])
