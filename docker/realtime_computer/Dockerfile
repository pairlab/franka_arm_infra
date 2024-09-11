FROM osrf/ros:noetic-desktop-full

ENV WS_DIR="/root"
WORKDIR ${WS_DIR}

SHELL ["/bin/bash", "-c"]

ARG DEBIAN_FRONTEND=noninteractive

ARG workstation_ip=192.168.0.1
ARG realtime_computer_ip=192.168.0.3
ARG robot_server_version=5

RUN mkdir /root/catkin_ws/src -p 
COPY ./csc376_prac3/prac3 /root/catkin_ws/src/prac3
COPY ./csc376_prac3/launch_robot_control_server.sh  /root/
# Dependencies for latency plot
RUN apt-get update && apt-get -y upgrade \
    && apt-get install -y \
    curl \
    grep \
    build-essential \
    make \
    libnuma-dev \
    python3 \
    python3-pip \    
    python3-distutils \
    gnuplot-qt\
    iproute2 \
    sshpass 
# Dependencies for Bash unit-tests
RUN apt-get update && apt-get install -y \
    bats \
    dialog \
    tmux \
    iputils-ping 

# Dependencies for franka-interface and protobuf
RUN apt-get update && apt-get install -y \
    git \
    vim \
    nano\
    wget \ 
    autoconf \ 
    automake \ 
    libtool \ 
    curl \ 
    make \
    g++ \
    unzip \
    python3-catkin-tools

RUN  apt-get install -y ros-noetic-moveit-core \
                    ros-noetic-moveit-ros-planning \
                    ros-noetic-moveit-ros-planning-interface \
                    ros-noetic-moveit-ros-perception \
                    ros-noetic-rviz-visual-tools \
                    ros-noetic-moveit-visual-tools \
                    ros-noetic-libfranka \
                    ros-noetic-franka-ros \
                    ros-noetic-ompl \
                    ros-noetic-moveit-planners \
                    ros-noetic-moveit-ros-visualization \
                    ros-noetic-moveit-simple-controller-manager \
                    ros-noetic-moveit-commander 

RUN cd ~/catkin_ws \
    && source /opt/ros/noetic/setup.bash \
    && rosdep install -y --from-paths . --ignore-src --rosdistro noetic \
    && sudo sh -c 'echo "deb http://packages.ros.org/ros-testing/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list' \
    && sudo apt update \
    && catkin config --extend /opt/ros/${ROS_DISTRO} --cmake-args -DCMAKE_BUILD_TYPE=Release \
    && catkin build

# Build and install Protobuf 
# RUN cd \
#     # && mkdir git \
#     && cd git \ 
#     && wget https://github.com/protocolbuffers/protobuf/releases/download/v3.11.4/protobuf-all-3.11.4.zip \ 
#     && unzip protobuf-all-3.11.4.zip \ 
#     && cd protobuf-3.11.4 \ 
#     && ./configure \
#     && make -j$(nproc) \
#     && make check -j$(nproc) \ 
#     && sudo make install \ 
#     && sudo ldconfig 




# # Dependencies for franka-interface and protobuf
# RUN apt-get install -y  ros-noetic-realsense2-camera ros-noetic-combined-robot-hw

# RUN cd ~/git \ 
#     && git clone --recurse-submodules https://github.com/Ruthrash/franka-interface \
#     && cd franka-interface 

# RUN echo "cloning libfranka"
# RUN cd ~/git/franka-interface \
#     && bash ./bash_scripts/clone_libfranka.sh ${robot_server_version}

# RUN echo "building libfranka"
# RUN cd ~/git/franka-interface \
#     && bash ./bash_scripts/make_libfranka.sh

# RUN echo "building franka-interface"
# RUN cd ~/git/franka-interface \
#     && bash ./bash_scripts/make_franka_interface.sh

# RUN echo "making ROS catkin workspace for franka-interface"
# RUN cd ~/git/franka-interface/catkin_ws/src \
#     && git clone https://github.com/frankaemika/franka_ros && cd franka_ros && git switch noetic-devel \
#     && rm -r franka_gazebo \ 
#     && cd ~/git/franka-interface \
#     && source /opt/ros/noetic/setup.bash \
#     && bash ./bash_scripts/make_catkin.sh 

# RUN echo "cloning and building libzmq"
# RUN cd ~/git \ 
#     && git clone https://github.com/zeromq/libzmq  \
#     && cd libzmq \ 
#     && mkdir build \ 
#     && cd build \ 
#     && cmake .. \ 
#     && sudo make  install 

# RUN echo "cloning and building cppzmq"
# RUN cd ~/git \ 
#     && git clone https://github.com/zeromq/cppzmq \
#     && cd cppzmq \ 
#     && mkdir build \ 
#     && cd build \ 
#     && cmake .. \ 
#     && sudo make  install 

RUN echo "source /opt/ros/noetic/setup.bash" >> ~/.bashrc
RUN echo "source ~/catkin_ws/devel/setup.bash " >> ~/.bashrc

RUN echo "export ROS_MASTER_URI=http://192.168.1.111:11311" >> ~/.bashrc
RUN echo "export ROS_IP=192.168.1.111" >> ~/.bashrc
# RUN echo "export ROS_IP=${realtime_computer_ip}" >> ~/.bashrc
# RUN echo "export ROS_MASTER_URI=http://${workstation_ip}:11311" >> ~/.bashrc

# RUN pip3 install zmq pyrealsense2 numba scipy   

ARG DEBIAN_FRONTEND=dialog

