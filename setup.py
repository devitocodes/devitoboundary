try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

setup(name='devitoboundary',
      use_scm_version=True,
      setup_requires=['setuptools_scm'],
      description="Immersed boundary tools for Devito codes",
      long_descritpion="""Devitoboundary is a ....""",
      url='http://www.devitoproject.org/',
      author="Imperial College London",
      author_email='opesci@imperial.ac.uk',
      license='MIT',
      packages=['devitoboundary'],
      install_requires=['numpy', 'simplejson'])
