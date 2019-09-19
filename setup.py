from setuptools import setup
from setuptools import find_packages
__version__ = '0.1.4'

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
                        'cherrypy>=7.1.0',
                        'six>=1.11.0',
                        'Pillow>=4.0.0',
                        'simplejson>=3.13.2']
)