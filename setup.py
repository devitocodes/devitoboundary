try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

setup(name='opesciboundary',
      use_scm_version=True,
      setup_requires=['setuptools_scm'],
      description="Immersed boundary tools for OPESCI codes",
      long_descritpion="""opesciboundary is a ....""",
      url='http://www.opesci.org/',
      author="Imperial College London",
      author_email='opesci@imperial.ac.uk',
      license='MIT',
      packages=['opesciboundary'],
      install_requires=['numpy', 'simplejson'])
