FROM quay.io/pypa/manylinux1_x86_64:latest
MAINTAINER Nathan Dunfield <nathan@dunfield.info>

RUN yum install -y nano tk mesa-libGLU-devel
RUN mkdir /sqlite; cd sqlite; \
    wget http://www.sqlite.org/src/tarball/SQLite-369f7498.tar.gz?uuid=369f74983bb1b6a6426260148018cdc084fc2b49; \    
    tar xfz SQL*; mkdir build; cd build; ../SQL*/configure; make install
RUN ln -s /opt/python/cp27-cp27mu/bin/python /bin/py27
RUN ln -s /opt/python/cp34-cp34m/bin/python  /bin/py34
RUN ln -s /opt/python/cp35-cp35m/bin/python  /bin/py35
RUN py27 -m pip install mercurial twine
RUN ln -s /opt/python/cp27-cp27mu/bin/hg /bin/hg
RUN mkdir /build
RUN py27 -m pip install --upgrade pip setuptools wheel cython sphinx
RUN py34 -m pip install --upgrade pip setuptools wheel cython sphinx
RUN py35 -m pip install --upgrade pip setuptools wheel cython sphinx
ADD bin /build/bin
RUN chmod +x /build/bin/*
WORKDIR /build
CMD ["/bin/bash"]