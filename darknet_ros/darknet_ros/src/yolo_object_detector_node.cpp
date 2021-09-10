/*
 * yolo_obstacle_detector_node.cpp
 *
 *  Created on: Dec 19, 2016
 *      Author: Marko Bjelonic
 *   Institute: ETH Zurich, Robotic Systems Lab
 */

#include <ros/ros.h>
#include <darknet_ros/YoloObjectDetector.hpp>

int main(int argc, char** argv) {
  ros::init(argc, argv, "darknet_ros");
  ros::NodeHandle nodeHandle("~");
  // Load Keras
  keras::KerasModel km("/home/ha/catkin_ws/src/team8-404-traffic-signs/darknet_ros/darknet_ros/src/dumped.nnet", true);

  darknet_ros::YoloObjectDetector yoloObjectDetector(nodeHandle, &km);

  ros::spin();
  return 0;
}
