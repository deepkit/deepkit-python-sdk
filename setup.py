from setuptools import setup
from setuptools import find_packages

__version__ = '1.0.2'

setup(name='deepkit',
      version=__version__,
      description='Python SDK for Deepkit',
      author='Marc J. Schmidt',
      author_email='marc@marcjschmidt.de',
      url='https://github.com/deepkit/deepkit-python-sdk',
      download_url='https://github.com/deepkit/deepkit-python-sdk/tarball/' + __version__,
      license='MIT',
      packages=find_packages(),
      install_requires=[
          'numpy',
          'Pillow>=4.0.0',
          'rx>=1.5',
          'typedload>=1.20',
          'PyYAML>=5.0.0',
          'psutil>=5.4.6',
          'websockets>=8.1'
      ],
      extras_require={
          'pytorch': ["torch"]
      })
