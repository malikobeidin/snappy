"""
Installation script for the snappy module.

Depends heavily on setuptools.
"""
no_setuptools_message = """
You need to have setuptools installed to build the snappy module, e.g. by:

  curl -O https://bitbucket.org/pypa/setuptools/raw/bootstrap/ez_setup.py
  sudo python ez_setup.py

or by installing the python-setuptools package (Debian/Ubuntu) or
python-setuptools-devel package (Fedora).  See

  https://pypi.python.org/pypi/setuptools

for more on setuptools.  
"""

old_setuptools_message = """
You need to have setuptools (>= 1.0) installed to build the snappy
module, e.g.

  sudo python -m easy_install "setuptools>=1.0"

or upgrading the python-setuptools package (Debian/Ubuntu) or
python-setuptools-devel package (Fedora). The latter one will only be
effective if setuptools was not installed with ez_setup.py or easy_install.
"""

no_cython_message = """
You need to have Cython (>= 0.11.2) installed to build the snappy
module since you're missing the autogenerated C/C++ files, e.g.

  sudo python -m easy_install "cython>=0.11.2"

"""

no_sphinx_message = """
You need to have Sphinx (>= 0.6.1) installed to rebuild the
documentation for snappy module, e.g.

  sudo python -m easy_install "sphinx>=0.6.1"

"""

try:
    import setuptools
    import pkg_resources
except ImportError:
    raise ImportError(no_setuptools_message)

# Make sure setuptools is installed in a late enough version

try:
    pkg_resources.working_set.require('setuptools>=1.0')
except (pkg_resources.DistributionNotFound, pkg_resources.VersionConflict):
    raise ImportError(old_setuptools_message)

# Remove '.' from the path so that Sphinx doesn't try to load the SnapPy module directly

import sys, os, glob, platform
try:
    sys.path.remove(os.path.realpath(os.curdir))
except:
    pass

from distutils.extension import Extension
from setuptools import setup, Command
from pkg_resources import load_entry_point

# A real clean

class clean(Command):
    user_options = []
    def initialize_options(self):
        pass 
    def finalize_options(self):
        pass
    def run(self):
        os.system('rm -rf build dist *.pyc')
        os.system('rm -rf snappy*.egg-info')
        os.system('rm -rf snappy/doc')
        os.system('rm -rf */Cy*.c */Cy*.h */Cy*.cpp')
        os.system('rm -rf SnapPy.c SnapPy.h SnapPyHP.cpp SnapPyHP.h')
        os.system('rm -rf hp_addl_code hp_kernel_code hp_unix_kit')
        for filename in os.listdir('hp_headers'):
            if filename != 'real_type.h':
                os.unlink(os.path.join('hp_headers', filename))

class build_docs(Command):
    user_options = []
    def initialize_options(self):
        pass 
    def finalize_options(self):
        pass
    def run(self):
        try:
            pkg_resources.working_set.require('sphinx>=0.6.1')
        except (pkg_resources.DistributionNotFound, pkg_resources.VersionConflict):
            raise ImportError(no_sphinx_message)
        sphinx_cmd = load_entry_point('Sphinx>=0.6.1', 'console_scripts', 'sphinx-build')
        sphinx_args = ['sphinx', '-a', '-E', '-d', 'doc-source/_build/doctrees',
                       'doc-source', 'snappy/doc']
        sphinx_cmd(sphinx_args)

# C source files we provide

base_code = glob.glob(os.path.join('kernel_code','*.c'))
unix_code = glob.glob(os.path.join('unix_kit','*.c'))
for unused in ['unix_UI.c', 'decode_new_DT.c']:
    file = os.path.join('unix_kit', unused)
    if file in unix_code:
        unix_code.remove(file)
addl_code = glob.glob(os.path.join('addl_code', '*.c')) + glob.glob(os.path.join('addl_code', '*.cc'))
code  =  base_code + unix_code + addl_code

# Symlinks for building the high precision version

def make_symlinks(source_files, target_dir):
    if sys.platform == 'win32':
        import shutil
        symlink = shutil.copyfile
    else:
        symlink = os.symlink
    if not os.path.exists(target_dir):
        os.mkdir(target_dir)
    for path in source_files:
        filename, ext = os.path.splitext(os.path.basename(path))
        new_ext = '.cpp' if ext == '.c' else ext
        link_name = os.path.join(target_dir, filename + new_ext)
        if sys.platform == 'win32':
            source = path
        else:
            source = os.path.join('..', path)
        if not os.path.exists(link_name):
            print('linking %s -> %s'%(link_name, source) )
            symlink(source, link_name)

def setup_symlinks():
    make_symlinks(base_code, 'hp_kernel_code')
    make_symlinks(unix_code, 'hp_unix_kit')                                 
    unix_headers = glob.glob(os.path.join('unix_kit', '*.h'))
    make_symlinks(unix_headers, 'hp_unix_kit')
    make_symlinks(addl_code, 'hp_addl_code')
    addl_headers = glob.glob(os.path.join('addl_code', '*.h'))
    make_symlinks(addl_headers, 'hp_addl_code')
    headers = glob.glob(os.path.join('headers', '*.h'))
    make_symlinks(headers, 'hp_headers')                                 

class build_symlinks(Command):
    user_options = []
    def initialize_options(self):
        pass 
    def finalize_options(self):
        pass
    def run(self):
        setup_symlinks()

# C++ source files we provide

if not os.path.exists('hp_kernel_code') and 'clean' not in sys.argv:
    setup_symlinks()

hp_base_code = glob.glob(os.path.join('hp_kernel_code','*.cpp'))
hp_unix_code = glob.glob(os.path.join('hp_unix_kit','*.cpp'))
hp_addl_code = glob.glob(os.path.join('hp_addl_code', '*.cpp'))# + glob.glob(os.path.join('addl_code', '*.cc'))
hp_qd_code = glob.glob(os.path.join('hp_qd', 'src', '*.cpp'))
hp_code  =  hp_base_code + hp_unix_code + hp_addl_code + hp_qd_code

# The SnapPy extension
SnapPyC = Extension(
    name = 'snappy.SnapPy',
    sources = ['SnapPy.c'] + code, 
    include_dirs = ['headers', 'unix_kit', 'addl_code'],
    extra_objects = [])

cython_sources = ['SnapPy.pyx']

# The high precision SnapPy extension
SnapPyHP = Extension(
    name = 'snappy.SnapPyHP',
    sources = ['SnapPyHP.cpp'] + hp_code, 
    include_dirs = ['hp_headers', 'hp_unix_kit', 'hp_addl_code', 'hp_qd/include'],
    language='c++',
    extra_compile_args = ['-msse2', '-mfpmath=sse', '-mieee-fp'],
    extra_objects = [])

cython_cpp_sources = ['SnapPyHP.pyx']

# The CyOpenGL extension
CyOpenGL_includes = ['.']
CyOpenGL_libs = []
CyOpenGL_extras = []
CyOpenGL_extra_link_args = []
if sys.platform == 'darwin':
    OS_X_ver = int(platform.mac_ver()[0].split('.')[1])
    if OS_X_ver > 7:
        path  =  '/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/' + \
                  'SDKs/MacOSX10.%d.sdk/System/Library/Frameworks/OpenGL.framework/Versions/A/Headers/' % OS_X_ver
    else:
       path =  '/System/Library/Frameworks/OpenGL.framework/Versions/Current/Headers/'
    CyOpenGL_includes += [path]
    CyOpenGL_extra_link_args = ['-framework', 'OpenGL']
elif sys.platform == 'linux2':
    CyOpenGL_includes += ['/usr/include/GL']
    CyOpenGL_libs += ['GL', 'GLU']
elif sys.platform == 'win32':
    CyOpenGL_includes += ['/mingw/include/GL']
    CyOpenGL_extras += ['/mingw/lib/libopengl32.a',
                        '/mingw/lib/libglu32.a']

cython_sources.append('opengl/CyOpenGL.pyx')

CyOpenGL = Extension(
    name = 'snappy.CyOpenGL',
    sources = ['opengl/CyOpenGL.c'], 
    include_dirs = CyOpenGL_includes,
    libraries = CyOpenGL_libs,
    extra_objects = CyOpenGL_extras,
    extra_link_args = CyOpenGL_extra_link_args)

# If have Cython, check that .c files are up to date:

try:
    from Cython.Build import cythonize
    if 'clean' not in sys.argv:
        cythonize(cython_sources)
        cythonize(cython_cpp_sources, language='c++')
except ImportError:
    for file in cython_sources:
        base = os.path.splitext(file)[0]
        if not os.path.exists(base + '.c'):
            raise ImportError(no_cython_message)
    for file in cython_cpp_sources:
        base = os.path.splitext(file)[0]
        if not os.path.exists(base + '.cpp'):
            raise ImportError(no_cython_message)
            

# Twister

twister_main_path = 'Twister/lib/'
twister_main_src = [twister_main_path + 'py_wrapper.cpp']
twister_kernel_path = twister_main_path + 'kernel/'
twister_kernel_src = [twister_kernel_path + file for file in
                      ['twister.cpp', 'manifold.cpp', 'parsing.cpp', 'global.cpp']]

TwisterCore = Extension(
	name = 'snappy.twister.twister_core',
	sources = twister_main_src + twister_kernel_src,
	include_dirs=[twister_kernel_path],
	language='c++' )

#ext_modules = [SnapPyC, TwisterCore]
#ext_modules = [SnapPyHP, TwisterCore]
ext_modules = [SnapPyC, SnapPyHP, TwisterCore]

try:
    import sage
    install_requires = ['plink>=1.7', 'ipython', 'pypng', 'spherogram>=1.3']
except ImportError:
    install_requires = ['plink>=1.7', 'ipython>=0.13', 'pypng', 'spherogram>=1.3', 'cypari>=1.0']
    if sys.platform == 'win32':
        install_requires.append('pyreadline>=2.0')

# Determine whether we will be able to activate the GUI code


try:
    if sys.version_info[0] < 3: 
        import Tkinter as Tk
    else:
        import tkinter as Tk
except ImportError:
    Tk = None

if Tk != None:
    if sys.version_info < (2,7): # ttk library is standard in Python 2.7 and newer
        install_requires.append('pyttk')   
    open_gl_headers = [CyOpenGL_includes[-1] + '/' + header for 
                       header in ['gl.h', 'glu.h']]
    if False in [os.path.exists(header) for header in open_gl_headers]:
        print("***WARNING***: OpenGL headers not found, not building CyOpenGL, will disable some graphics features. ")
    else:
        ext_modules.append(CyOpenGL)
else:
    print("***WARNING**: Tkinter not installed, GUI won't work")
    
# Get version number:
exec(open('snappy/version.py').read())

# Get long description from README
long_description = open('README').read()
long_description = long_description.split('==\n\n')[1]
long_description = long_description.split('Credits')[0]

# Off we go ...
setup( name = 'snappy',
       version = version,
       zip_safe = False,
       install_requires = install_requires,
       dependency_links = ['http://www.math.uic.edu/t3m/plink/',
                           'http://www.math.uic.edu/t3m/SnapPy-nest'],
       packages = ['snappy', 'snappy/manifolds', 'snappy/twister',
                   'snappy/snap', 'snappy/snap/t3mlite', 'snappy/ptolemy'],
       package_data = {
        'snappy' : ['togl/*-tk*/Togl2.0/*',
                    'togl/*-tk*/Togl2.1/*',
                    'togl/*-tk*/mactoolbar*/*',
                    'info_icon.gif', 'SnapPy.ico',
                    'doc/*.*',
                    'doc/_images/*',
                    'doc/_sources/*',
                    'doc/_static/*'],
        'snappy/manifolds' : ['manifolds.sqlite',
                              'more_manifolds.sqlite',
                              'HTWKnots/*.gz'],
        'snappy/twister' : ['surfaces/*'],
        'snappy/ptolemy':['magma/*.magma_template',
                          'testing_files/*magma_out.bz2',
                          'regina_testing_files/*magma_out.bz2',
                          'testing_files_generalized/*magma_out.bz2',
                          'regina_testing_files_generalized/*magma_out.bz2',
                          'testing_files_rur/*rur.bz2'],
        },
       package_dir = {'snappy/twister':'Twister/lib'},
       ext_modules = ext_modules,
       cmdclass =  {'clean' : clean,
                    'build_symlinks': build_symlinks,
                    'build_docs': build_docs},
       entry_points = {'console_scripts': ['SnapPy = snappy.app:main']},

       description= 'Studying the topology and geometry of 3-manifolds, with a focus on hyperbolic structures.', 
       long_description = long_description,
       author = 'Marc Culler and Nathan M. Dunfield',
       author_email = 'culler@uic.edu, nathan@dunfield.info',
       license='GPLv2+',
       url = 'http://snappy.computop.org',
       classifiers = [
           'Development Status :: 5 - Production/Stable',
           'Intended Audience :: Science/Research',
           'License :: OSI Approved :: GNU General Public License v2 or later (GPLv2+)',
           'Operating System :: OS Independent',
           'Programming Language :: C',
           'Programming Language :: C++', 
           'Programming Language :: Python',
           'Programming Language :: Cython',
           'Topic :: Scientific/Engineering :: Mathematics',
        ],
        keywords = '3-manifolds, topology, hyperbolic geometry',
)
