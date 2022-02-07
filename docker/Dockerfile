# syntax=docker/dockerfile:1.2
FROM ubuntu:20.04

# Install OS-level packages
COPY docker/install_packages.sh docker/install_shfmt.sh docker/timer.sh docker/install_fonts.sh  /usr/local/bin/
RUN --mount=type=cache,target=/var/cache/apt \
    --mount=type=cache,target=/var/lib/apt \
    install_packages.sh python3.9 python3.9-venv git shellcheck curl
RUN install_shfmt.sh
RUN install_fonts.sh

# Create and activate virtual environment
ENV VIRTUAL_ENV=/root/.venv
RUN python3.9 -m venv "$VIRTUAL_ENV"
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Setup config
COPY config /root/.config

# Setup root home directory
WORKDIR /root/sundry_musings

# Install Python dependencies
COPY requirements/requirements.txt "$VIRTUAL_ENV"
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --upgrade pip setuptools wheel && \
    pip install --requirement "$VIRTUAL_ENV"/requirements.txt

# Copy over source code and install using pip
COPY src src
COPY pyproject.toml setup.cfg ./
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --editable .

CMD [ "timer.sh" ]
