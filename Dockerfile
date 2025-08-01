#####################################
# RCP CaaS requirement (Image)
#####################################
# The best practice is to use an image
# with GPU support pre-built by Nvidia.
# https://catalog.ngc.nvidia.com/orgs/nvidia/containers/

# For example, if you want to use an image with pytorch already installed
# FROM nvcr.io/nvidia/pytorch:23.11-py3

# TUTORIAL ONLY
# In this example we'll use a smaller image to speed up the build process.
# Basic image based on ubuntu 22.04
FROM pytorch/pytorch:2.2.0-cuda11.8-cudnn8-runtime

ENV PYOPENGL_PLATFORM=egl

# Install required packages
RUN apt-get update && apt-get install -y \
    python3-pip \
    freeglut3-dev \
    libgl1-mesa-glx \
    mesa-utils \
    xvfb \
    && rm -rf /var/lib/apt/lists/*


#####################################
# RCP CaaS requirement (Storage)
#####################################
# Create your user inside the container.
# This block is needed to correctly map
# your EPFL user id inside the container.
# Without this mapping, you are not able
# to access files from the external storage.
ARG LDAP_USERNAME
ARG LDAP_UID
ARG LDAP_GROUPNAME
ARG LDAP_GID
RUN groupadd ${LDAP_GROUPNAME} --gid ${LDAP_GID}
RUN useradd -m -s /bin/bash -g ${LDAP_GROUPNAME} -u ${LDAP_UID} ${LDAP_USERNAME}
#####################################

RUN mkdir -p /home/${LDAP_USERNAME}
WORKDIR /home/${LDAP_USERNAME}
COPY requirements.txt /tmp/requirements.txt

# Install additional dependencies
RUN pip install --no-cache-dir -r /tmp/requirements.txt

#install own Miniworld
COPY ./Miniworld /app/Miniworld
WORKDIR /app/Miniworld
RUN pip install -e .
WORKDIR /home/${LDAP_USERNAME}
# Copy your code inside the container
COPY ./ /home/${LDAP_USERNAME}

# Set your user as owner of the new copied files
RUN chown -R ${LDAP_USERNAME}:${LDAP_GROUPNAME} /home/${LDAP_USERNAME}

# Set the working directory in your user's home
WORKDIR /home/${LDAP_USERNAME}
USER ${LDAP_USERNAME}

ENV PYGLET_HEADLESS=True
