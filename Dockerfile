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

RUN apt-get install ninja-build

# RUN git clone https://github.com/llvm/llvm-project.git
# WORKDIR /llvm-project
# RUN git checkout llvmorg-16.0.5
# RUN cmake -S llvm -B build -G Ninja -DLLVM_ENABLE_PROJECTS="clang;lld" -DCMAKE_BUILD_TYPE=Debug
# RUN cmake --build build -j8
# RUN sudo cmake --install build --prefix /opt
# RUN sudo ninja -C build check-llvm 
# WORKDIR /

RUN wget https://github.com/llvm/llvm-project/releases/download/llvmorg-16.0.4/clang+llvm-16.0.4-x86_64-linux-gnu-ubuntu-22.04.tar.xz
RUN mkdir /opt/llvm 
RUN mv clang+llvm-16.0.4-x86_64-linux-gnu-ubuntu-22.04.tar.xz /opt/llvm 
WORKDIR /opt/llvm 
RUN tar -xJf clang+llvm-16.0.4-x86_64-linux-gnu-ubuntu-22.04.tar.xz
RUN rm clang+llvm-16.0.4-x86_64-linux-gnu-ubuntu-22.04.tar.xz
WORKDIR /
ENV LLVM_DIR=/opt/llvm/clang+llvm-16.0.4-x86_64-linux-gnu-ubuntu-22.04
ENV LLVM_CONFIG=${LLVM_DIR}/bin/llvm-config
ENV PATH=${PATH}:${LLVM_DIR}/bin


# Set user specific environment variables
ENV USER root
ENV HOME /root
# Switch to user
USER ${USER}


# jupyter notebook port
# EXPOSE 8888



