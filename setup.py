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

class SnapPyTest(Command):
    user_options = []
    def initialize_options(self):
        pass 
    def finalize_options(self):
        pass
    def run(self):
        build_lib_dir = os.path.join(
            'build',
            'lib.{platform}-{version_info[0]}.{version_info[1]}'.format(
                platform=sysconfig.get_platform(),
                version_info=sys.version_info)
        )
        sys.path.insert(0, build_lib_dir)
        from snappy.test import runtests
        print('Running tests ...')
        sys.exit(runtests())

if sys.platform == 'win32':
    pythons = [
        r'C:\Python27\python.exe',
        r'C:\Python27-x64\python.exe',
        r'C:\Python34\python.exe',
        r'C:\Python34-x64\python.exe',
        ]
elif sys.platform == 'darwin':
    pythons = [
        'python2.7',
        'python3.4',
        'python3.5',
        'python3.6',
        ]
elif site.__file__.startswith('/opt/python/cp'):
    pythons = [
        'python2.7',
        'python3.4',
        'python3.5',
        'python3.6',
        ]
else:
    pythons = [
        'python2.7',
        'python3.5'
    ]

class SnapPyRelease(Command):
    # The -rX option modifies the wheel name by adding rcX to the version string.
    # This is for uploading to testpypi, which will not allow uploading two
    # wheels with the same name.
    user_options = [('rctag=', 'r', 'index for rc tag to be appended to version (e.g. -r2 -> rc2)')]
    def initialize_options(self):
        self.rctag = None
    def finalize_options(self):
        if self.rctag:
            self.rctag = 'rc%s'%self.rctag
    def run(self):
        if os.path.exists('build'):
            shutil.rmtree('build')
        if os.path.exists('dist'):
            shutil.rmtree('dist')
        for python in pythons:
            try:
                subprocess.check_call([python, 'setup.py', 'build'])
            except subprocess.CalledProcessError:
                raise RuntimeError('Build failed for %s.'%python)
                sys.exit(1)
            try:
                subprocess.check_call([python, 'setup.py', 'build_docs'])
            except subprocess.CalledProcessError:
                raise RuntimeError('Documentation build failed for %s.'%python)
                sys.exit(1)
            try:
                subprocess.check_call([python, 'setup.py', 'test'])
            except subprocess.CalledProcessError:
                raise RuntimeError('Test failed for %s.'%python)
                sys.exit(1)
            try:
                subprocess.check_call([python, 'setup.py', 'bdist_wheel'])
            except subprocess.CalledProcessError:
                raise RuntimeError('Error building wheel for %s.'%python)
        if sys.platform.startswith('linux'):
            # auditwheel generates names with more tags than allowed by pypi
            extra_tag = re.compile('linux_x86_64\.|linux_i686\.')
            # build wheels tagged as manylinux1
            for wheelname in [name for name in os.listdir('dist') if name.endswith('.whl')]:
                original_path = os.path.join('dist', wheelname)
                subprocess.check_call(['auditwheel', 'addtag', '-w', 'dist', original_path])
                os.remove(original_path)
        else:
            extra_tag = None
        version_tag = re.compile('-([^-]*)-')
        for wheel_name in [name for name in os.listdir('dist') if name.endswith('.whl')]:
            new_name = wheel_name
            if extra_tag:
                new_name = extra_tag.sub('', new_name, 1)
            if self.rctag:
                new_name = version_tag.sub('-\g<1>%s-'%self.rctag, new_name, 1)
            os.rename(os.path.join('dist', wheel_name), os.path.join('dist', new_name))
        try:
            subprocess.check_call(['python', 'setup.py', 'sdist'])
        except subprocess.CalledProcessError:
            raise RuntimeError('Error building sdist archive for %s.'%python)
        sdist_version = re.compile('-([^-]*)(.tar.gz)|-([^-]*)(.zip)')
        for archive_name in [name for name in os.listdir('dist')
                             if name.endswith('tar.gz') or name.endswith('.zip')]:
            if self.rctag:
                new_name = sdist_version.sub('-\g<1>%s\g<2>'%self.rctag, archive_name, 1)
                os.rename(os.path.join('dist', archive_name), os.path.join('dist', new_name))

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

try:
    cc = distutils.ccompiler.get_default_compiler()
except AttributeError:
    cc = None
for arg in sys.argv:
    if arg.startswith('--compiler='):
        cc = arg.split('=')[1]

# The SnapPy extension
snappy_extra_compile_args = []
snappy_extra_link_args = []
if sys.platform == 'win32' and cc == 'msvc':
    snappy_extra_compile_args.append('/EHsc')
    snappy_extra_link_args.append('-lmsvcr90')

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
    if sys.maxsize > 2**32:
        hp_extra_compile_args = []
    else:
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
        try:
            from setuptools import msvc9_support
            include_dirs = msvc9_support.query_vcvarsall(9.0)['include'].split(';')
        except ImportError:
            from setuptools import msvc
            include_dirs = msvc.msvc9_query_vcvarsall(9.0)['include'].split(';')
    
        GL_include_dirs = [os.path.join(path, 'gl') for path in include_dirs
                           if path.upper().find('WINSDK')>0]
        CyOpenGL_includes += GL_include_dirs
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
    extra_link_args = CyOpenGL_extra_link_args)

# If have Cython, check that .c files are up to date:

try:
    from Cython.Build import cythonize
    if 'clean' not in sys.argv:
        cython_sources = [file for file in cython_sources if os.path.exists(file)]
        cythonize(cython_sources)
        cython_cpp_sources = [file for file in cython_cpp_sources if os.path.exists(file)]
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
        install_requires.append('ipython>=0.13,<5.0')
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
            results = [os.path.exists(os.path.join(path, header)) for path in CyOpenGL_includes]
            missing[header] = (True in results)
        if False in missing.values():
            print("***WARNING***: OpenGL headers not found, not building CyOpenGL, will disable some graphics features. ")
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
                   'snappy/verify', 'snappy/dev', 'snappy/dev/peripheral'],
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
                      'snappy/dev':'dev/extended_ptolemy',
                      'snappy/dev/peripheral':'dev/extended_ptolemy/peripheral', 
                  }, 
       ext_modules = ext_modules,
       cmdclass =  {'clean' : SnapPyClean,
                    'build_docs': SnapPyBuildDocs,
                    'test': SnapPyTest,
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
