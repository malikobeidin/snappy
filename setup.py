"""
Installation script for the snappy module.

Depends heavily on setuptools.
"""
no_setuptools_message = """ 
You need to have setuptools installed to build the snappy module. See:

  https://packaging.python.org/installing/

"""

no_cython_message = """
You need to have Cython (>= 0.26) installed to build the snappy
module since you're missing the autogenerated C/C++ files, e.g.

  sudo python -m pip install "cython>=0.26"

"""

no_sphinx_message = """
You need to have Sphinx (>= 1.3) installed to rebuild the
documentation for snappy module, e.g.

  sudo python -m pip install "sphinx>=1.3"

"""
import os, platform, shutil, site, subprocess, sys, sysconfig, re
from os.path import getmtime, exists
from distutils.util import get_platform
from distutils.ccompiler import get_default_compiler
from glob import glob

# Xcode deprecated libstdc++ in OSX 10.9 and removed it in 10.14.
# Also, it stopped providing i386 libraries in 10.14.  So if we are
# building on a system newer than 10.13 we are pretty much forced to
# build for 64 bits only, with a minimum target of 10.9, and to use
# libc++.

if sys.platform == 'darwin':
    minor_os_version = int(platform.mac_ver()[0].split('.')[1])
    if minor_os_version > 13:
        macOS_compile_args = macOS_link_args = ['-stdlib=libc++', '-mmacosx-version-min=10.9']
    else:
        macOS_compile_args = macOS_link_args = ['-mmacosx-version-min=10.6']

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
        sphinx_args = ['-a', '-E', '-d', 'doc_src/_build/doctrees',
                       'doc_src', 'python/doc']
        import sphinx
        if sphinx.version_info < (1, 7):
            sphinx_args = ['sphinx'] + sphinx_args
        sphinx_cmd(sphinx_args)

def distutils_dir_name(dname):
    """Returns the name of a distutils build subdirectory"""
    name = "build/{prefix}.{plat}-{ver[0]}.{ver[1]}".format(
        prefix=dname, plat=sysconfig.get_platform(), ver=sys.version_info)
    if dname == 'temp' and sys.platform == 'win32':
        name += os.sep + 'Release'
    return name
        
def build_lib_dir():
    return os.path.abspath(distutils_dir_name('lib'))

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

# For manylinux1 wheels, need to set the platform name manually to
# avoid having to "repair" the wheels later. 
try:
    from wheel.bdist_wheel import bdist_wheel
    class SnapPyBuildWheel(bdist_wheel):
        def initialize_options(self):
            bdist_wheel.initialize_options(self)
            if sys.platform.startswith('linux'):
                plat = get_platform().replace('linux', 'manylinux1')
                plat = plat.replace('-', '_')
                self.plat_name = plat
except ImportError:
    SnapPyBuildWheel = None
    
    
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
                check_call([python, 'setup.py', 'bdist_egg'])
            if self.install:
                check_call([python, 'setup.py', 'pip_install'])
            else:
                check_call([python, 'setup.py', 'bdist_wheel'])


        # Build sdist using the *first* specified Python
        check_call([pythons[0], 'setup.py', 'sdist'])

        # Double-check the Linux wheels
        if sys.platform.startswith('linux'):
            for name in os.listdir('dist'):
                if name.endswith('.whl'):
                    subprocess.check_call(['auditwheel', 'repair', os.path.join('dist', name)])

class SnapPyPipInstall(Command):
    user_options = []
    def initialize_options(self):
        pass
    def finalize_options(self):
        pass
    def run(self):
        python = sys.executable
        check_call([python, 'setup.py', 'bdist_wheel'])
        egginfo = 'snappy.egg-info'
        if os.path.exists(egginfo):
            shutil.rmtree(egginfo)
        wheels = glob('dist' + os.sep + '*.whl')
        new_wheel = max(wheels, key=os.path.getmtime)            
        check_call([python, '-m', 'pip', 'install', '--upgrade',
                    '--upgrade-strategy', 'only-if-needed',
                    new_wheel])
                    
# C source files we provide

base_code = glob(os.path.join('kernel', 'kernel_code','*.c'))
unix_code = glob(os.path.join('kernel', 'unix_kit','*.c'))
for unused in ['unix_UI.c', 'decode_new_DT.c']:
    file = os.path.join('kernel', 'unix_kit', unused)
    if file in unix_code:
        unix_code.remove(file)
addl_code = glob(os.path.join('kernel', 'addl_code', '*.c'))
addl_code += glob(os.path.join('kernel', 'addl_code', '*.cpp'))
code  =  base_code + unix_code + addl_code

# C++ source files we provide

hp_qd_code = glob(os.path.join('quad_double', 'qd', 'src', '*.cpp'))

# These are the Cython files that directly get compiled

cython_sources = ['cython/SnapPy.pyx', 'opengl/CyOpenGL.pyx']
cython_cpp_sources = ['cython/SnapPyHP.pyx']

# This is the complete list of Cython files, including those included
# by the above.

all_cython_files = cython_sources + cython_cpp_sources
all_cython_files += ['cython/SnapPycore.pxi', 'cython/SnapPy.pxi']
all_cython_files += glob(os.path.join('cython','core', '*.pyx'))

# If we have Cython, regenerate .c files as needed:
try:
    from Cython.Build import cythonize
    if 'clean' not in sys.argv:
        cython_sources = [file for file in cython_sources if exists(file)]
        cythonize(cython_sources,
                  compiler_directives={'embedsignature': True})
        cython_cpp_sources = [file for file in cython_cpp_sources if exists(file)]
        cythonize(cython_cpp_sources, language='c++',
                  compiler_directives={'embedsignature': True})
except ImportError:
    for file in cython_sources:
        base = os.path.splitext(file)[0]
        if not exists(base + '.c'):
            raise ImportError(no_cython_message)
    for file in cython_cpp_sources:
        base = os.path.splitext(file)[0]
        if not exists(base + '.cpp'):
            raise ImportError(no_cython_message)

# We check manually which object files need to be rebuilt; distutils
# is overly cautious and always rebuilds everything, which makes
# development painful.

def replace_ext(file, new_ext):
    root, ext = os.path.splitext(file)
    return root + '.' + new_ext

def modtime(file):
    if os.path.exists(file):
        return os.path.getmtime(file)
    else:
        return 0.0

class SourceAndObjectFiles(object):
    def __init__(self):
        self.sources_to_build, self.up_to_date_objects = [], []
        self.temp_dir = distutils_dir_name('temp')
        self.obj_ext = 'obj' if sys.platform.startswith('win') else 'o'

    def add(self, source_file, dependency_mod_time=0.0):
        object_file = self.temp_dir + os.sep + replace_ext(source_file, self.obj_ext)
        if modtime(object_file) < max(modtime(source_file), dependency_mod_time):
            self.sources_to_build.append(source_file)
        else:
            self.up_to_date_objects.append(object_file)

snappy_ext_files = SourceAndObjectFiles()
hp_snappy_ext_files = SourceAndObjectFiles()
cy_source_mod_time = max([modtime('cython' + os.sep + file)
                          for file in all_cython_files])
snappy_ext_files.add('cython' + os.sep + 'SnapPy.c', cy_source_mod_time)
hp_snappy_ext_files.add('cython' + os.sep + 'SnapPyHP.cpp', cy_source_mod_time)
                         
for file in code:
    snappy_ext_files.add(file)
    hp_file = 'quad_double' + replace_ext(file, 'cpp')[len('kernel'):]
    assert os.path.exists(hp_file)
    hp_snappy_ext_files.add(hp_file, modtime(file))

for hp_file in hp_qd_code:
    hp_snappy_ext_files.add(hp_file)

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
if sys.platform == 'darwin':
    snappy_extra_compile_args += macOS_compile_args
    snappy_extra_link_args += macOS_link_args

SnapPyC = Extension(
    name = 'snappy.SnapPy',
    sources = snappy_ext_files.sources_to_build, 
    include_dirs = ['kernel/headers', 'kernel/unix_kit',
                    'kernel/addl_code', 'kernel/real_type'],
    language='c++',
    extra_compile_args=snappy_extra_compile_args,
    extra_link_args=snappy_extra_link_args,
    extra_objects = snappy_ext_files.up_to_date_objects)


# The high precision SnapPy extension

hp_extra_link_args = []
if sys.platform == 'win32' and cc == 'msvc':
    hp_extra_compile_args = []
    if platform.architecture()[0] == '32bit':
        hp_extra_compile_args = ['/arch:SSE2']
    hp_extra_compile_args.append('/EHsc')
else:
    hp_extra_compile_args = ['-msse2', '-mfpmath=sse', '-mieee-fp']
if sys.platform == 'darwin':
    hp_extra_compile_args += macOS_compile_args
    hp_extra_link_args += macOS_link_args

# SnapPyHP depends implicitly on the source for the main kernel, so we 
# we delete certain object files to force distutils to rebuild them.

if len(hp_snappy_ext_files.sources_to_build):
    ldir = distutils_dir_name('lib')
    matches = glob(os.path.join(ldir, 'snappy', 'SnapPyHP.*'))
    if len(matches) > 0:
        os.remove(matches[0])
    
SnapPyHP = Extension(
    name = 'snappy.SnapPyHP',
    sources = hp_snappy_ext_files.sources_to_build,
    include_dirs = ['kernel/headers', 'kernel/unix_kit',
                    'kernel/addl_code', 'kernel/kernel_code',
                    'quad_double/real_type', 'quad_double/qd/include'],
    language='c++',
    extra_compile_args = hp_extra_compile_args,
    extra_link_args = hp_extra_link_args,
    extra_objects = hp_snappy_ext_files.up_to_date_objects)


# The CyOpenGL extension
CyOpenGL_includes = ['.']
CyOpenGL_libs = []
CyOpenGL_extras = []
CyOpenGL_extra_link_args = []
if sys.platform == 'darwin':
    OS_X_ver = int(platform.mac_ver()[0].split('.')[1])
    sdk_roots = ['/Library/Developer/CommandLineTools/SDKs',
                 '/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs']
    header_dir = '/MacOSX10.%d.sdk/System/Library/Frameworks/OpenGL.framework/Versions/Current/Headers/' % OS_X_ver
    poss_includes = [root + header_dir for root in sdk_roots]
    poss_includes += ['/System/Library/Frameworks/OpenGL.framework/Versions/Current/Headers/']
    CyOpenGL_includes += [path for path in poss_includes if os.path.exists(path)][:1]
    CyOpenGL_extra_link_args = ['-framework', 'OpenGL']
    CyOpenGL_extra_link_args += macOS_link_args

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


CyOpenGL = Extension(
    name = 'snappy.CyOpenGL',
    sources = ['opengl/CyOpenGL.c'], 
    include_dirs = CyOpenGL_includes,
    libraries = CyOpenGL_libs,
    extra_objects = CyOpenGL_extras,
    extra_link_args = CyOpenGL_extra_link_args,
    language='c++'
)


# Twister

twister_main_path = 'twister/lib/'
twister_main_src = [twister_main_path + 'py_wrapper.cpp']
twister_kernel_path = twister_main_path + 'kernel/'
twister_kernel_src = [twister_kernel_path + file for file in
                      ['twister.cpp', 'manifold.cpp', 'parsing.cpp', 'global.cpp']]
twister_extra_compile_args = []
twister_extra_link_args = []
if sys.platform == 'win32' and cc == 'msvc':
    twister_extra_compile_args.append('/EHsc')
if sys.platform == 'darwin':
    twister_extra_compile_args += macOS_compile_args
    twister_extra_link_args += macOS_compile_args

TwisterCore = Extension(
    name = 'snappy.twister.twister_core',
    sources = twister_main_src + twister_kernel_src,
    include_dirs=[twister_kernel_path],
    extra_compile_args=twister_extra_compile_args,
    extra_link_args=twister_extra_link_args,
    language='c++' )

ext_modules = [SnapPyC, SnapPyHP, TwisterCore]

install_requires = ['plink>=2.2', 'spherogram>=1.8.1', 'FXrays>=1.3',
                    'pypng', 'decorator', 'future', 'snappy_manifolds>=1.0']
try:
    import sage
except ImportError:
    install_requires.append('cypari>=2.2')
    install_requires.append('ipython>=0.13')
    if sys.version_info < (3, 4):
        install_requires.append('ipython<6.0')
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
long_description = open('README.rst').read()
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
           'snappy' : ['info_icon.gif', 'SnapPy.ico', 'SnapPy.png',
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
                    'bdist_wheel':SnapPyBuildWheel,
                    'pip_install':SnapPyPipInstall,
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
