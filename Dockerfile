FROM rust:1.71 AS rust_base
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04 AS cuda_base

FROM cuda_base AS merged_base
COPY --from=rust_base /usr/local/cargo /usr/local/cargo
COPY --from=rust_base /usr/local/rustup /usr/local/rustup

ENV PATH="/usr/local/cargo/bin:${PATH}"
ENV RUSTUP_HOME="/usr/local/rustup"
ENV CARGO_HOME="/usr/local/cargo"
ENV DEBIAN_FRONTEND noninteractive

RUN rustup component add rustfmt

RUN apt-get update && \
    apt-get install -y lsb-release

RUN apt-get update && apt-get -y install cmake 

RUN apt-get update && apt-get install -y python3.10 python3-pip

RUN apt-get update && apt-get install -y vim wget sudo software-properties-common gnupg git 

RUN apt-get update && apt-get install ninja-build

RUN apt-get update && apt-get install -y cuda-nsight-systems-11-8

RUN apt-get update && apt-get install -y zip 

RUN pip install numpy scipy matplotlib

# This step builds and installs Clang+LLVM toolchain from source that matches the version the 
# Rust compiler uses. Openmp is also enabled.
# LLVM gets installed in opt/llvm
# RUN mkdir /opt/llvm 
# RUN git clone https://github.com/llvm/llvm-project.git
# WORKDIR /llvm-project 
# RUN git checkout llvmorg-16.0.5 
# RUN cmake -S llvm -B build -G Ninja \
#     -DLLVM_ENABLE_PROJECTS="clang;lld" -DLLVM_ENABLE_RUNTIMES="openmp" -DCMAKE_BUILD_TYPE=Release \
#     -DCMAKE_INSTALL_PREFIX=/opt/llvm
# RUN ninja -C build install

RUN wget https://www.agner.org/optimize/asmlib.zip
RUN mkdir /opt/asmlib
RUN unzip asmlib.zip -d /opt/asmlib
RUN rm /asmlib.zip

WORKDIR /
RUN rm -rf /llvm-project
ENV LLVM_DIR=/opt/llvm
ENV LLVM_CONFIG=${LLVM_DIR}/bin/llvm-config
ENV PATH=${PATH}:${LLVM_DIR}/bin

# Download FFTW for reverts for the GCC compiler
RUN wget https://www.fftw.org/fftw-3.3.10.tar.gz
RUN tar -zxvf fftw-3.3.10.tar.gz
RUN rm fftw-3.3.10.tar.gz
WORKDIR /fftw-3.3.10

# FFTW install for float precision with openmp support
RUN ./configure --enable-float --enable-openmp 
# RUN make CC=/opt/llvm/bin/clang -j8 
RUN make -j8
RUN make install 

# FFTW install for double precision with openmp support
RUN make clean 
RUN ./configure --enable-openmp
# RUN make CC=/opt/llvm/bin/clang -j8 
RUN make -j8
RUN make install 

# FFTW install for long double precision with openmp support
RUN make clean 
RUN ./configure --enable-long-double --enable-openmp
# RUN make CC=/opt/llvm/bin/clang -j8 
RUN make -j8
RUN make install

WORKDIR /
RUN rm -rf /fftw-3.3.10
# Set user specific environment variables
ENV USER root
ENV HOME /root
# Switch to user
USER ${USER}


# jupyter notebook port
# EXPOSE 8888



