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

# This step installs a pre-built Clang+LLVM toolchain that matches the version the 
# Rust compiler uses.

# RUN wget https://github.com/llvm/llvm-project/releases/download/llvmorg-16.0.4/clang+llvm-16.0.4-x86_64-linux-gnu-ubuntu-22.04.tar.xz 
# RUN mv clang+llvm-16.0.4-x86_64-linux-gnu-ubuntu-22.04.tar.xz /opt 
# WORKDIR /opt 
# RUN tar -xJf clang+llvm-16.0.4-x86_64-linux-gnu-ubuntu-22.04.tar.xz && mv clang+llvm-16.0.4-x86_64-linux-gnu-ubuntu-22.04 llvm_prebuilt
# RUN rm clang+llvm-16.0.4-x86_64-linux-gnu-ubuntu-22.04.tar.xz
# WORKDIR /

RUN mkdir /opt/llvm 
RUN git clone https://github.com/llvm/llvm-project.git
WORKDIR /llvm-project 
RUN git checkout llvmorg-16.0.4 
RUN cmake -S llvm -B build -G Ninja \
    -DLLVM_ENABLE_PROJECTS="clang;lld" -DLLVM_ENABLE_RUNTIMES="openmp" -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX=/opt/llvm
# RUN cmake --build build -j8 
RUN ninja -C build install

WORKDIR /
RUN rm -rf /llvm-project
ENV LLVM_DIR=/opt/llvm
ENV LLVM_CONFIG=${LLVM_DIR}/bin/llvm-config
ENV PATH=${PATH}:${LLVM_DIR}/bin

# Install FFTW for reverts for the Clang compiler
RUN wget https://www.fftw.org/fftw-3.3.10.tar.gz
RUN tar -zxvf fftw-3.3.10.tar.gz
RUN rm fftw-3.3.10.tar.gz
WORKDIR /fftw-3.3.10

RUN ./configure --enable-float --enable-openmp 
RUN make CC=/opt/llvm/bin/clang -j8 
RUN make install 

RUN make clean 
RUN ./configure --enable-openmp
RUN make CC=/opt/llvm/bin/clang -j8 
RUN make install 

RUN make clean 
RUN ./configure --enable-long-double --enable-openmp
RUN make CC=/opt/llvm/bin/clang -j8 
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



