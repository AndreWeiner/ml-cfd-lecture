FROM ubuntu:26.04
ARG DEBIAN_FRONTEND=noninteractive
ARG OPENFOAM_VERSION=2606
ARG TORCH_VERSION=2.14.1
ARG LIBTORCH_SHA256=0f02329a5266153cffa9dc12926ff2221fa626207b894d40bd61905cc2079821
RUN apt-get update && apt-get install --no-install-recommends -y \
    ca-certificates cmake build-essential unzip wget git
RUN wget -q -O /tmp/add-openfoam-repo.sh https://dl.openfoam.com/add-debian-repo.sh \
    && bash /tmp/add-openfoam-repo.sh && rm /tmp/add-openfoam-repo.sh
RUN apt-get update && apt-get install --no-install-recommends -y \
    openfoam${OPENFOAM_VERSION}-dev && rm -rf /var/lib/apt/lists/*
ENV FOAM_PATH=/usr/lib/openfoam/openfoam${OPENFOAM_VERSION}
RUN echo ". ${FOAM_PATH}/etc/bashrc" >> /etc/bash.bashrc
RUN wget -q -O /tmp/libtorch.zip \
    "https://download.pytorch.org/libtorch/cpu/libtorch-shared-with-deps-${TORCH_VERSION}%2Bcpu.zip" \
    && if [ -n "$LIBTORCH_SHA256" ]; then echo "$LIBTORCH_SHA256  /tmp/libtorch.zip" | sha256sum -c -; fi \
    && unzip -q /tmp/libtorch.zip -d /opt && rm /tmp/libtorch.zip
ENV TORCH_LIBRARIES=/opt/libtorch
COPY test /opt/of-pytorch-tests
RUN bash -c 'source "$FOAM_PATH/etc/bashrc" && /opt/of-pytorch-tests/run-tests.sh'
