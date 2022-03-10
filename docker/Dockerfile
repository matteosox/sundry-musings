# syntax=docker/dockerfile:1.2
FROM condaforge/mambaforge:4.11.0-4 as build

# Create Conda environment
COPY requirements/conda-linux-64.lock docker/create_conda_env.sh ./
RUN ./create_conda_env.sh

FROM ubuntu:20.04 as runtime

# Install OS-level packages
COPY docker/install_fonts.sh docker/install_packages.sh docker/install_shfmt.sh docker/timer.sh /usr/local/bin/
RUN --mount=type=cache,target=/var/cache/apt \
    --mount=type=cache,target=/var/lib/apt \
    install_packages.sh git shellcheck curl ca-certificates tini
RUN install_shfmt.sh
RUN install_fonts.sh

# Move to home directory
WORKDIR /root

# Make RUN commands use the new environment:
RUN echo "source ~/.venv/bin/activate" | cat - ~/.bashrc > /tmp/out && mv /tmp/out ~/.bashrc
SHELL ["/bin/bash", "--login", "-c"]

# Install conda env
COPY --from=build /venv .venv

# Setup config
COPY config .config

# Install Python dependencies
COPY requirements/requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --upgrade pip setuptools wheel && \
    pip install --requirement requirements.txt

# Move to repo directory
WORKDIR /root/sundry_musings

# Copy over source code and install using pip
COPY src src
COPY pyproject.toml setup.cfg ./
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --editable .

ENTRYPOINT ["tini", "-v", "--"]
CMD [ "timer.sh" ]
