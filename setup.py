try:
    from setuptools import setup
except ImportError:
    from ez_setup import use_setuptools
    use_setuptools()
from setuptools import find_packages, Extension, setup  # noqa: F811
import numpy.distutils.misc_util
import os

include_dirs = (
    ['/usr/local/include'] +
    numpy.distutils.misc_util.get_numpy_include_dirs())
sources = [
    '_seq_lda.cpp', 'alpha.cpp', 'lda.cpp', 'markov_lda.cpp',
    'callback_lda.cpp', 'suff_stats.cpp', 'utils.cpp',
    'variational_inference.cpp']
sources = [os.path.join('seq_lda/_seq_lda/', f) for f in sources]

seq_lda = Extension('seq_lda._seq_lda',
                    define_macros=[('MAJOR_VERSION', '1'),
                                   ('MINOR_VERSION', '0')],
                    include_dirs=include_dirs,
                    libraries=[],
                    library_dirs=['/usr/local/lib'],
                    sources=sources,
                    extra_compile_args=['--std=c++11 -Wno-sign-compare'],
                    language='c++')

setup(name='seq_lda',
      version='1.0',
      description='Latent Dirichlet Allocation for sequential data.',
      author='Eric Crawford',
      author_email='eric.crawford@mail.mcgill.ca',
      packages=find_packages(),
      url='',
      long_description='''
A package for using LDA for multitask sequence learning.
LDA implementation based on lda-c by David Blei.
''',
      ext_modules=[seq_lda])
