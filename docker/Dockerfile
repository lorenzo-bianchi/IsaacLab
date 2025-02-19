# Base image
FROM nvcr.io/nvidia/isaac-sim:4.2.0

ENV DEBIAN_FRONTEND noninteractive

# # Create internal users group
# RUN groupadd -r internal

# Configure environment variables to expose Nvidia GPUs
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=all

# Install necessary dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    apt-utils \
    build-essential \
    curl \
    fuse \
    g++ \
    gcc \
    git \
    gnupg2 \
    htop \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    locales \
    lsb-release \
    make \
    nano \
    ninja-build \
    python-dev-is-python3 \
    python-is-python3 \
    python3 \
    python3-argcomplete \
    python3-autopep8 \
    python3-dev \
    python3-numpy \
    python3-pip \
    python3-pygments \
    python3-pytest-pylint \
    python3-vcstools \
    screen \
    software-properties-common \
    sudo \
    uncrustify \
    wget \
    whois \
    zsh \
    zsh-doc && \
    add-apt-repository universe && \
    apt-get autoremove -y && \
    apt-get autoclean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*/apt/lists/*

# Configure Git to accept different ownerships for local repository clones
RUN git config --system --add safe.directory '*'

# # Create a Python virtual environment and install packages in it
# WORKDIR /opt
# RUN --mount=type=bind,source=requirements.txt,target=/opt/requirements.txt --mount=type=bind,source=requirements_dev.txt,target=/opt/requirements_dev.txt \
#   python3 -m venv --system-site-packages dua-venv && \
#   . dua-venv/bin/activate && \
#   pip install -U -r requirements.txt -r requirements_dev.txt && \
#   chgrp -R internal /opt/dua-venv && \
#   chmod -R g+rw /opt/dua-venv
# WORKDIR /root

# Configure language and locale
RUN locale-gen en_US.UTF-8 && \
    update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8
ENV LANG=en_US.UTF-8
    
# Install ROS 2 Humble Hawksbill desktop
# RUN curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key  -o /usr/share/keyrings/ros-archive-keyring.gpg
# RUN echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu jammy main" | tee /etc/apt/sources.list.d/ros2.list > /dev/null
# RUN apt-get update && apt-get install -y --no-install-recommends \
#     python3-colcon-argcomplete \
#     python3-colcon-common-extensions \
#     python3-colcon-mixin \
#     python3-rosdep \
#     ros-humble-ament-lint \
#     ros-humble-desktop \
#     ros-humble-diagnostic-msgs \
#     ros-humble-diagnostic-updater \
#     ros-humble-eigen3-cmake-module \
#     ros-humble-joint-state-publisher \
#     ros-humble-robot-state-publisher \
#     ros-humble-ros-ign-bridge \
#     ros-humble-xacro && \
#     rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*/apt/lists/*

# Configure ROS Middleware implementation to use
ENV RMW_IMPLEMENTATION=rmw_fastrtps_cpp

# Set environment variables
ENV ACCEPT_EULA=Y
ENV PRIVACY_CONSENT=Y
ENV LC_ALL=en_US.UTF-8
ENV ROS_DISTRO=humble
ENV ROS_VERSION=2
ENV ROS_PYTHON_VERSION=3

# Set up volumes for caches and data
VOLUME ["/isaac-sim/kit/cache", "/root/.cache/ov", "/root/.cache/pip", "/root/.cache/nvidia/GLCache", "/root/.nv/ComputeCache", "/root/.nvidia-omniverse/logs", "/root/.local/share/ov/data", "/root/Documents"]

##########################################

ARG USER_UID=1000

RUN userdel -r ubuntu || true
#   mkpasswd -m sha-512 PASSWORD duatemplate
RUN useradd -r -m -s /usr/bin/zsh -u ${USER_UID} -G adm,dialout,internal,plugdev,sudo,tty,video -p '$6$x54Ia175iLtd9S3A$dbgitjeJSXvLatutbDg75lOfW/A75O4MbUb7MiQGR1uYJLgqn8no54QVIO.nSEqPVQYd3YvtThU1hccJTn/vt/' neo
ENV HOME=/home/neo

# Create workspace directory: host workspaces will be mounted here
RUN mkdir ${HOME}/workspace && \
    chown neo:neo ${HOME}/workspace
  
# Create directory for shell history file
RUN mkdir ${HOME}/zsh_history && \
    chown neo:neo ${HOME}/zsh_history

# Create SSH directory for user
RUN mkdir ${HOME}/.ssh

# Switch to internal user
USER neo
WORKDIR ${HOME}

# Copy user configuration files
COPY --chown=neo:neo ./aliases.sh ./.aliases.sh
COPY --chown=neo:neo ./bashrc ./.bashrc
COPY --chown=neo:neo ./commands.sh ./.commands.sh
COPY --chown=neo:neo ./nanorc ./.nanorc
COPY --chown=neo:neo ./vimrc ./.vimrc

# Configure Zsh for internal user
ENV ZSH=${HOME}/.oh-my-zsh
ENV ZSH_CUSTOM=${ZSH}/custom
ENV ZSH_PLUGINS=${ZSH_CUSTOM}/plugins
ENV ZSH_THEMES=${ZSH_CUSTOM}/themes
RUN wget -qO- https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh | zsh || true
RUN git clone --single-branch --branch 'master' --depth 1 https://github.com/zsh-users/zsh-syntax-highlighting.git ${ZSH_PLUGINS}/zsh-syntax-highlighting \
    && git clone --single-branch --branch 'master' --depth 1 https://github.com/zsh-users/zsh-autosuggestions ${ZSH_PLUGINS}/zsh-autosuggestions \
    && git clone --single-branch --depth 1 https://github.com/romkatv/powerlevel10k.git ${ZSH_THEMES}/powerlevel10k
COPY --chown=neo:neo ./p10k.zsh ./.p10k.zsh
COPY --chown=neo:neo ./zshrc ./.zshrc

ENV DEBIAN_FRONTEND=dialog
CMD ["bash"]
