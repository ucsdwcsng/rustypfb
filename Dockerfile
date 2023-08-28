FROM rust:1.67.0 AS rust_base
FROM nvidia/cuda:11.7.1-cudnn8-devel-ubuntu20.04 AS cuda_base

FROM cuda_base AS merged_base
COPY --from=rust_base /usr/local/cargo /usr/local/cargo
COPY --from=rust_base /usr/local/rustup /usr/local/rustup

ENV PATH="/usr/local/cargo/bin:${PATH}"
ENV RUSTUP_HOME="/usr/local/rustup"
ENV CARGO_HOME="/usr/local/cargo"

RUN rustup component add rustfmt

RUN apt-get update && \
    apt-get install -y lsb-release

RUN apt-get update && apt-get install -y python3.8 python3-pip

RUN apt-get update && apt-get install -y vim

# Set user specific environment variables
ENV USER root
ENV HOME /root
# Switch to user
USER ${USER}


# jupyter notebook port
# EXPOSE 8888


