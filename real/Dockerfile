from gibsonchallenge/baseline_sac:castro

ARG DEBIAN_FRONTEND=noninteractive

# setup timezone
RUN echo 'Etc/UTC' > /etc/timezone && \
    ln -s /usr/share/zoneinfo/Etc/UTC /etc/localtime && \
    apt-get update && apt-get install -q -y tzdata && rm -rf /var/lib/apt/lists/*

# install packages
RUN apt-get update && apt-get install -q -y \
    dirmngr \
    gnupg2 \
    && rm -rf /var/lib/apt/lists/*

# setup keys
RUN apt-key adv --keyserver hkp://keyserver.ubuntu.com:80 --recv-keys C1CF6E31E6BADE8868B172B4F42ED6FBAB17C654

# setup sources.list
RUN echo "deb http://packages.ros.org/ros/ubuntu bionic main" > /etc/apt/sources.list.d/ros1-latest.list

# install bootstrap tools
RUN apt-get update && apt-get install --no-install-recommends -y \
    python-rosdep \
    python-rosinstall \
    python-vcstools \
    && rm -rf /var/lib/apt/lists/*

# setup environment
ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8

ENV ROS_DISTRO melodic
# bootstrap rosdep
RUN rosdep init && \
  rosdep update --rosdistro $ROS_DISTRO

# install ros packages
RUN apt-get update && apt-get install -y \
    ros-melodic-ros-core=1.4.1-0* \
    && rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y \
    ros-melodic-ros-base=1.4.1-0* \
    && rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y \
    ros-melodic-robot=1.4.1-0* \
    && rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y \
    ros-melodic-perception=1.4.1-0* \
    && rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y \
    ros-melodic-desktop=1.4.1-0* \
    && rm -rf /var/lib/apt/lists/*

#RUN apt-get update && apt-get install -y \
#    ros-melodic-desktop-full=1.4.1-0* \
#    && rm -rf /var/lib/apt/lists/*

RUN conda update -y conda

RUN conda create -y -n py2 python=2.7.17

ENV PATH /miniconda/envs/py2/bin:/usr/local/nvidia/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin

COPY challenge_real.py /challenge_real.py

WORKDIR /miniconda/bin

#RUN /bin/bash -c "source /miniconda/bin/activate"
#RUN /bin/bash -c "source activate /miniconda/envs/py2"
#RUN /bin/bash -c "source /opt/ros/melodic/setup.bash" 
RUN pip install pytest
RUN pip install tensorflow-gpu==1.15.0

WORKDIR /opt/agents
RUN pip install -e .

RUN pip install IPython
RUN pip install pyYAML 
RUN pip install rospkg
RUN pip install opencv-python

WORKDIR /
RUN sed -i 's/gibson2.envs.challenge/challenge_real/g' agent.py
