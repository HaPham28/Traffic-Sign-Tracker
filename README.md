# Traffic Sign Tracker
This project is to create a Traffic Sign Tracker to detect 4 classes of traffic signs: Stop, Yield, Speed Limit and Pedestrian Crossing.
The program can also read the numbers of all speed limit signs with 2 digit.
The program is tested on a VM with ROS-kinetic(2016)

## Installation
This repository assumes that the Robot Operating System (ROS-kinetic 2016) has already been installed on your environment

## Running
To set up and run the program, please follow the steps below:

1) On your VM, open the terminal and type the following command lines:

```bash
$ mkdir -p catkin_ws/src && cd catkin_ws/src
$ catkin_init_workspace
$ cd ../
$ catkin_make
$ cd src
$ git clone https://github.tamu.edu/autodrive-common/team8-404-traffic-signs.git
```
2) Download provided weights file from https://drive.google.com/file/d/1RJGxCSKvLlnsmcUkRTh_6GLQVMAUEwre/view?usp=sharing and put file in the following repository:
darknet_ros/darknet_ros/yolo_network_config/weights

3) Make pkg with the following commands:
$ cd catkin_ws
$ catkin_make

4) Run the following commands in separate terminals:
Terminal 1:
$ roscore
Terminal 2:
$ rostopic echo objects 
Terminal 3:
$ source ~/catkin_ws/devel/setup.bash
$ roslaunch darknet_ros yolov4-tiny-obj.launch
