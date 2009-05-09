"""
This is a setup.py script generated by py2applet

Usage:
    python setup.py py2app
"""

from setuptools import setup, Command
import os

class clean(Command):
    user_options = []
    def initialize_options(self):
        pass 
    def finalize_options(self):
        pass
    def run(self):
        os.system("rm -rf build dist *.pyc")

APP = ['SnapPython.py']
DATA_FILES = []
OPTIONS = {'argv_emulation': True,
 'excludes': 'scipy,numpy',
 'packages': 'SnapPy,IPython',
 'includes': 'gzip,tarfile,readline',
}

setup(
    app=APP,
    data_files=DATA_FILES,
    options={'py2app': OPTIONS},
    setup_requires=['py2app'],
    cmdclass   = {'clean' : clean},
)
