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

no_cython_message = """
You need to have Cython (>= 0.11.2) installed to build the snappy
module since you're missing the autogenerated C/C++ files, e.g.

  sudo python -m easy_install "cython>=0.11.2"

"""

no_sphinx_message = """
You need to have Sphinx (>= 1.3) installed to rebuild the
documentation for snappy module, e.g.

  sudo python -m easy_install "sphinx>=1.3"

"""
import os, platform, shutil, site, subprocess, sys, sysconfig
from os.path import getmtime, exists
from distutils.util import get_platform
from distutils.ccompiler import get_default_compiler
from glob import glob

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

from setuptools import distutils

# Remove '.' from the path so that Sphinx doesn't try to load the SnapPy module directly

try:
    sys.path.remove(os.path.realpath(os.curdir))
except:
    pass

from distutils.extension import Extension
from setuptools import setup, Command
from pkg_resources import load_entry_point

# A real clean

class SnapPyClean(Command):
    user_options = []
    def initialize_options(self):
        pass 
    def finalize_options(self):
        pass
    def run(self):
        junkdirs = (glob('build/lib*') +
                    glob('build/bdist*') +
                    glob('build/temp*') +
                    glob('snappy*.egg-info') +
                    ['__pycache__', os.path.join('python', 'doc')]
        )
        for dir in junkdirs:
            try:
                shutil.rmtree(dir)
            except OSError:
                pass
        junkfiles = glob('python/*.so*') + glob('python/*.pyc') 
        for generated in ['SnapPy.c', 'SnapPy.h', 'SnapPyHP.cpp', 'SnapPyHP.h']:
            junkfiles.append(os.path.join('cython', generated))
        for file in junkfiles:
            try:
                os.remove(file)
            except OSError:
                pass

class SnapPyBuildDocs(Command):
    user_options = []
    def initialize_options(self):
        pass 
    def finalize_options(self):
        pass
    def run(self):
        try:
            pkg_resources.working_set.require('sphinx>=1.3')
        except (pkg_resources.DistributionNotFound, pkg_resources.VersionConflict):
            raise ImportError(no_sphinx_message)
        sphinx_cmd = load_entry_point('Sphinx>=1.3', 'console_scripts', 'sphinx-build')
        sphinx_args = ['sphinx', '-a', '-E', '-d', 'doc_src/_build/doctrees',
                       'doc_src', 'python/doc']
        sphinx_cmd(sphinx_args)

def build_lib_dir():
    return os.path.abspath(os.path.join(
        'build',
        'lib.{platform}-{version_info[0]}.{version_info[1]}'.format(
            platform=get_platform(),
            version_info=sys.version_info)
    ))

class SnapPyBuildAll(Command):
    user_options = []
    def initialize_options(self):
        pass 
    def finalize_options(self):
        pass
    def run(self):
        python = sys.executable
        subprocess.call([python, 'setup.py', 'build'])
        subprocess.call([python, 'setup.py', 'build_docs'])
        subprocess.call([python, 'setup.py', 'build'])

class SnapPyTest(Command):
    user_options = []
    def initialize_options(self):
        pass 
    def finalize_options(self):
        pass
    def run(self):
        sys.path.insert(0, build_lib_dir())
        from snappy.test import runtests
        print('Running tests ...')
        sys.exit(runtests())

class SnapPyApp(Command):
    user_options = []
    def initialize_options(self):
        pass 
    def finalize_options(self):
        pass
    def run(self):
        sys.path.insert(0, build_lib_dir())
        import snappy.app
        snappy.app.main()

def check_call(args):
    try:
        subprocess.check_call(args)
    except subprocess.CalledProcessError:
        executable = args[0]
        command = [a for a in args if not a.startswith('-')][-1]
        raise RuntimeError(command + ' failed for ' + executable)
    
class SnapPyRelease(Command):
    user_options = [('install', 'i', 'install the release into each Python')]
    def initialize_options(self):
        self.install = False
    def finalize_options(self):
        pass
    def run(self):
        if exists('build'):
            shutil.rmtree('build')
        if exists('dist'):
            shutil.rmtree('dist')

        pythons = os.environ.get('RELEASE_PYTHONS', sys.executable).split(',')
        for python in pythons:
            check_call([python, 'setup.py', 'build_all'])
            check_call([python, 'setup.py', 'test'])
            if sys.platform.startswith('linux'):
                plat = get_platform().replace('linux', 'manylinux1')
                plat = plat.replace('-', '_')
                check_call([python, 'setup.py', 'bdist_wheel', '-p', plat])
                check_call([python, 'setup.py', 'bdist_egg'])
            else:
                check_call([python, 'setup.py', 'bdist_wheel'])
            if self.install:
                check_call([python, 'setup.py', 'install'])

        # Build sdist using the *first* specified Python
        check_call([pythons[0], 'setup.py', 'sdist'])

        # Double-check the Linux wheels
        if sys.platform.startswith('linux'):
            for name in os.listdir('dist'):
                if name.endswith('.whl'):
                    subprocess.check_call(['auditwheel', 'repair', os.path.join('dist', name)])

# C source files we provide

base_code = glob(os.path.join('kernel', 'kernel_code','*.c'))
unix_code = glob(os.path.join('kernel', 'unix_kit','*.c'))
for unused in ['unix_UI.c', 'decode_new_DT.c']:
    file = os.path.join('kernel', 'unix_kit', unused)
    if file in unix_code:
        unix_code.remove(file)
addl_code = glob(os.path.join('kernel', 'addl_code', '*.c')) + glob(os.path.join('kernel', 'addl_code', '*.cc'))
code  =  base_code + unix_code + addl_code

# C++ source files we provide

hp_base_code = glob(os.path.join('quad_double', 'kernel_code','*.cpp'))
hp_unix_code = glob(os.path.join('quad_double', 'unix_kit','*.cpp'))
hp_addl_code = glob(os.path.join('quad_double', 'addl_code', '*.cpp'))
hp_qd_code = glob(os.path.join('quad_double', 'qd', 'src', '*.cpp'))
hp_code  =  hp_base_code + hp_unix_code + hp_addl_code + hp_qd_code

# The compiler we will be using

cc = get_default_compiler()
for arg in sys.argv:
    if arg.startswith('--compiler='):
        cc = arg.split('=')[1]

# The SnapPy extension
snappy_extra_compile_args = []
snappy_extra_link_args = []
if sys.platform == 'win32':
    if cc == 'msvc':
        snappy_extra_compile_args.append('/EHsc')
    else:
        if sys.version_info.major == 2:
            snappy_extra_link_args.append('-lmsvcr90')
        elif sys.version_info == (3,4):
            snappy_extra_link_args.append('-lmsvcr100')
            
SnapPyC = Extension(
    name = 'snappy.SnapPy',
    sources = ['cython/SnapPy.c'] + code, 
    include_dirs = ['kernel/headers', 'kernel/unix_kit', 'kernel/addl_code', 'kernel/real_type'],
    language='c++',
    extra_compile_args=snappy_extra_compile_args,
    extra_link_args=snappy_extra_link_args,
    extra_objects = [])

cython_sources = ['cython/SnapPy.pyx']

if sys.platform == 'win32' and cc == 'msvc':
    hp_extra_compile_args = []
    if platform.architecture()[0] == '32bit':
        hp_extra_compile_args = ['/arch:SSE2']
    hp_extra_compile_args += ['/EHsc']
else:
    hp_extra_compile_args = ['-msse2', '-mfpmath=sse', '-mieee-fp']

# The high precision SnapPy extension
SnapPyHP = Extension(
    name = 'snappy.SnapPyHP',
    sources = ['cython/SnapPyHP.cpp'] + hp_code, 
    include_dirs = ['kernel/headers', 'kernel/unix_kit', 'kernel/addl_code', 'kernel/kernel_code',
                    'quad_double/real_type', 'quad_double/qd/include'],
    language='c++',
    extra_compile_args = hp_extra_compile_args,
    extra_objects = [])

cython_cpp_sources = ['cython/SnapPyHP.pyx']

# The CyOpenGL extension
CyOpenGL_includes = ['.']
CyOpenGL_libs = []
CyOpenGL_extras = []
CyOpenGL_extra_link_args = []
if sys.platform == 'darwin':
    OS_X_ver = int(platform.mac_ver()[0].split('.')[1])
    if OS_X_ver > 7:
        path  = '/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/' + \
                'SDKs/MacOSX10.%d.sdk/System/Library/Frameworks/OpenGL.framework/Versions/Current/Headers/' % OS_X_ver
        CyOpenGL_includes += [path]
    CyOpenGL_includes += ['/System/Library/Frameworks/OpenGL.framework/Versions/Current/Headers/']
    CyOpenGL_extra_link_args = ['-framework', 'OpenGL']
elif sys.platform == 'linux2' or sys.platform == 'linux':
    CyOpenGL_includes += ['/usr/include/GL']
    CyOpenGL_libs += ['GL']
elif sys.platform == 'win32':
    if cc == 'msvc':
        include_dirs = []
        if sys.version_info.major == 2:
            from setuptools.msvc import msvc9_query_vcvarsall
            includes = msvc9_query_vcvarsall(9.0)['include'].split(';')
            include_dirs += includes
            include_dirs += [os.path.join(path, 'gl') for path in includes]
        elif sys.version_info.major == 3 and sys.version_info.minor == 4:
            from distutils.msvc9compiler import query_vcvarsall
            includes = query_vcvarsall(10.0)['include'].split(';')
            include_dirs += includes
            if sys.maxsize <= 2**32:
                include_dirs += [os.path.join(path, 'gl') for path in includes]
        elif sys.version_info.major == 3 and sys.version_info.minor > 4:
            from distutils.msvc9compiler import query_vcvarsall
            includes = query_vcvarsall(14.0)['include'].split(';')
            include_dirs += includes
            include_dirs += [os.path.join(path, 'gl') for path in includes]
        CyOpenGL_includes += include_dirs
        CyOpenGL_extras += ['opengl32.lib']
    else:
        CyOpenGL_includes += ['/mingw/include/GL']
        CyOpenGL_extras += ['/mingw/lib/libopengl32.a']

cython_sources.append('opengl/CyOpenGL.pyx')

CyOpenGL = Extension(
    name = 'snappy.CyOpenGL',
    sources = ['opengl/CyOpenGL.c'], 
    include_dirs = CyOpenGL_includes,
    libraries = CyOpenGL_libs,
    extra_objects = CyOpenGL_extras,
    extra_link_args = CyOpenGL_extra_link_args,
    language='c++'
)

# Check whether CyOpenGL.c is up to date.  If not we will need to
# patch it on Windows.
cyopengl_c = os.path.join('opengl', 'CyOpenGL.c')
cyopengl_pyx = os.path.join('opengl', 'CyOpenGL.pyx')
if exists(cyopengl_c):
    cyopengl_c_rebuilt = (getmtime(cyopengl_c) < getmtime(cyopengl_pyx))
else:
    cyopengl_c_rebuilt = True

 # If we have Cython, regenerate .c files as needed:
try:
    from Cython.Build import cythonize
    if 'clean' not in sys.argv:
        cython_sources = [file for file in cython_sources if exists(file)]
        cythonize(cython_sources)
        cython_cpp_sources = [file for file in cython_cpp_sources if exists(file)]
        cythonize(cython_cpp_sources, language='c++')
except ImportError:
    for file in cython_sources:
        base = os.path.splitext(file)[0]
        if not exists(base + '.c'):
            raise ImportError(no_cython_message)
    for file in cython_cpp_sources:
        base = os.path.splitext(file)[0]
        if not exists(base + '.cpp'):
            raise ImportError(no_cython_message)
            
# Patch up CyOpenGL.c for Windows. (This assumes sed is available)
# As of version 0.25.2 Cython assumes that 1L is a 64-bit constant,
# which happens to be false on Windows.
if sys.platform == 'win32' and cyopengl_c_rebuilt:
    subprocess.call(['sed', '-i',  '-e s/1L<<53/1LL<<53/', cyopengl_c])

# Twister

twister_main_path = 'twister/lib/'
twister_main_src = [twister_main_path + 'py_wrapper.cpp']
twister_kernel_path = twister_main_path + 'kernel/'
twister_kernel_src = [twister_kernel_path + file for file in
                      ['twister.cpp', 'manifold.cpp', 'parsing.cpp', 'global.cpp']]
twister_extra_compile_args = []
if sys.platform == 'win32' and cc == 'msvc':
    twister_extra_compile_args.append('/EHsc')

TwisterCore = Extension(
    name = 'snappy.twister.twister_core',
    sources = twister_main_src + twister_kernel_src,
    include_dirs=[twister_kernel_path],
    extra_compile_args=twister_extra_compile_args,
    language='c++' )

ext_modules = [SnapPyC, SnapPyHP, TwisterCore]

install_requires = ['plink>=1.9.1', 'spherogram>=1.5a1', 'FXrays>=1.3',
                    'pypng', 'decorator', 'future']
try:
    import sage
except ImportError:
    install_requires.append('cypari>=1.2.2')
    if sys.version_info < (2,7):  # Newer IPythons only support Python 2.7
        install_requires.append('ipython>=0.13,<2.0')
    else:
        install_requires.append('ipython>=0.13')
        # As of 2016-10-12 iPython 5 imports enum but does not require it.
        # install_requires.append('enum>=0.4.6')
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
    if sys.platform == 'win32': # really only for Visual C++
        ext_modules.append(CyOpenGL)
    else:
        missing = {}
        for header in ['gl.h']:
            results = [exists(os.path.join(path, header))
                       for path in CyOpenGL_includes]
            missing[header] = (True in results)
        if False in missing.values():
            print("***WARNING***: OpenGL headers not found, "
                  "not building CyOpenGL, "
                  "will disable some graphics features.")
        else:
            ext_modules.append(CyOpenGL)
else:
    print("***WARNING**: Tkinter not installed, GUI won't work")
    
# Get version number:
exec(open('python/version.py').read())

# Get long description from README
long_description = open('README').read()
long_description = long_description.split('==\n\n')[1]
long_description = long_description.split('Credits')[0]

# Off we go ...
setup( name = 'snappy',
       version = version,
       zip_safe = False,
       install_requires = install_requires,
       packages = ['snappy', 'snappy/manifolds', 'snappy/twister',
                   'snappy/snap', 'snappy/snap/t3mlite', 'snappy/ptolemy',
                   'snappy/verify', 'snappy/dev', 'snappy/dev/peripheral',
                   'snappy/togl',
       ],
       package_data = {
           'snappy' : ['info_icon.gif', 'SnapPy.ico',
                       'doc/*.*',
                       'doc/_images/*',
                       'doc/_sources/*',
                       'doc/_static/*'],
           'snappy/togl': ['*-tk*/Togl2.0/*',
                       '*-tk*/Togl2.1/*',
                       '*-tk*/mactoolbar*/*'],
           'snappy/manifolds' : ['manifolds.sqlite',
                                 'more_manifolds.sqlite',
                                 'platonic_manifolds.sqlite',
                                 'HTWKnots/*.gz'],
           'snappy/twister' : ['surfaces/*'],
           'snappy/ptolemy':['magma/*.magma_template',
                             'testing_files/*magma_out.bz2',
                             'testing_files/data/pgl2/OrientableCuspedCensus/03_tetrahedra/*magma_out',
                             'regina_testing_files/*magma_out.bz2',
                             'testing_files_generalized/*magma_out.bz2',
                             'regina_testing_files_generalized/*magma_out.bz2',
                             'testing_files_rur/*rur.bz2']
       },
       package_dir = {'snappy':'python', 'snappy/manifolds':'python/manifolds',
                      'snappy/twister':'twister/lib',  'snappy/snap':'python/snap',
                      'snappy/snap/t3mlite':'python/snap/t3mlite',
                      'snappy/ptolemy':'python/ptolemy',
                      'snappy/verify':'python/verify',
                      'snappy/togl': 'python/togl',
                      'snappy/dev':'dev/extended_ptolemy',
                      'snappy/dev/peripheral':'dev/extended_ptolemy/peripheral', 
                  }, 
       ext_modules = ext_modules,
       cmdclass =  {'clean' : SnapPyClean,
                    'build_docs': SnapPyBuildDocs,
                    'build_all': SnapPyBuildAll,
                    'test': SnapPyTest,
                    'app': SnapPyApp,
                    'release': SnapPyRelease,
       },
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
