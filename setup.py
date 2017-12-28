import os
import io
from setuptools import setup, find_packages
from setuptools.command.install import install as _install


def readfile(fname):
    path = os.path.join(os.path.dirname(__file__), fname)
    return io.open(path, encoding='utf8').read()


class Install(_install):
    def run(self):
        _install.do_egg_install(self)


setup(name='DAPPER',
      version='0.1',
      description='Python implementation of the Dynamic Author-Persona Topic Model using fast Conjugate Computations.',
      long_description=readfile('README.md'),
      # url='https://github.com/robert-giaquinto/dapper',
      author='Robert Giaquinto',
      license='MIT',
      packages=find_packages(),
      include_package_data=True,
      cmdclass={'install': Install},
      install_requires=['numpy', 'scipy'],
      setup_requires=['numpy', 'scipy'],
      zip_safe=False)
