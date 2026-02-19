FROM osrf/ros:noetic-desktop-full

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install --no-install-recommends -y \
    vim \
    nano \
    wget \
    doxygen \
    graphviz \
    build-essential \
    libeigen3-dev \
    python3-catkin-tools \
    ros-noetic-dynamic-reconfigure \
    ros-noetic-tf2-ros \
    ros-noetic-navigation \
    x11-apps \
    firefox \
    git \
    unzip \
    python3-pip  \
    python3-tk \
    && rm -rf /var/lib/apt/lists/* \
    && mkdir -p /catkin_build_ws/src \
    && mkdir -p /catkin_make_ws/src

WORKDIR /catkin_build_ws
RUN /bin/bash -c "source /opt/ros/noetic/setup.sh; rosdep init && rosdep update; \
                  source /opt/ros/noetic/setup.sh; catkin init; catkin config --extend /opt/ros/noetic; \
                  source /opt/ros/noetic/setup.sh; catkin build;"


WORKDIR /catkin_make_ws
RUN /bin/bash -c "source /opt/ros/noetic/setup.sh; catkin_make;"

RUN useradd -m  -s /bin/bash ubuntu
RUN usermod -aG sudo ubuntu && echo "ubuntu ALL=(ALL) NOPASSWD: ALL" > /etc/sudoers.d/ubuntu
RUN chmod 044 /etc/sudoers.d/ubuntu
USER ubuntu:ubuntu
WORKDIR /home/ubuntu
RUN echo "source /opt/ros/noetic/setup.bash" >> /home/ubuntu/.bashrc

# create catkin ws and copy files
RUN mkdir -p /home/ubuntu/simulation_ws/src
WORKDIR /home/ubuntu/simulation_ws/src
COPY my_turtlebot3_openai_example.zip .
RUN unzip -q my_turtlebot3_openai_example.zip
RUN rm -rf my_turtlebot3_openai_example.zip
COPY openai_ros.zip .
RUN unzip -q openai_ros.zip
RUN rm -rf openai_ros.zip
COPY turtlebot3.zip .
RUN unzip -q turtlebot3.zip
RUN rm -rf turtlebot3.zip
WORKDIR /home/ubuntu/simulation_ws/src/my_turtlebot3_openai_example/scripts
RUN chmod +x start_deepqlearning.py
RUN chmod +x start_qlearning_v2.py

# install Python dependencies
RUN python3 -m pip install gym==0.25.0
RUN python3 -m pip install GitPython
RUN python3 -m pip install --upgrade pip
RUN python3 -m pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cu121 torch

WORKDIR /home/ubuntu/simulation_ws
RUN /bin/bash -c "source /opt/ros/noetic/setup.bash; catkin init; catkin build; source devel/setup.bash"
RUN echo "source /home/ubuntu/simulation_ws/devel/setup.bash" >> /home/ubuntu/.bashrc
RUN echo "export TURTLEBOT3_MODEL=burger" >> /home/ubuntu/.bashrc

ENTRYPOINT ["/bin/bash"]