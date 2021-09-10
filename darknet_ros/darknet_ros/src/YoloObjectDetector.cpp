/*
 * YoloObjectDetector.cpp
 *
 *  Created on: Dec 19, 2016
 *      Author: Marko Bjelonic
 *   Institute: ETH Zurich, Robotic Systems Lab
 */

#include <iostream>

// yolo object detector
#include "darknet_ros/YoloObjectDetector.hpp"

// Check for xServer
#include <X11/Xlib.h>

#ifdef DARKNET_FILE_PATH
std::string darknetFilePath_ = DARKNET_FILE_PATH;
#else
#error Path of darknet repository is not defined in CMakeLists.txt.
#endif

namespace darknet_ros {

char* cfg;
char* weights;
char* data;
char** detectionNames;

YoloObjectDetector::YoloObjectDetector(ros::NodeHandle nh, keras::KerasModel *km)
    : nodeHandle_(nh), km(km), imageTransport_(nodeHandle_), numClasses_(0), classLabels_(0), rosBoxes_(0), rosBoxCounter_(0) {
  ROS_INFO("[YoloObjectDetector] Node started.");

  // Read parameters from config file.
  if (!readParameters()) {
    ros::requestShutdown();
  }

  init();
}

YoloObjectDetector::~YoloObjectDetector() {
  {
    boost::unique_lock<boost::shared_mutex> lockNodeStatus(mutexNodeStatus_);
    isNodeRunning_ = false;
  }
  yoloThread_.join();
}

bool YoloObjectDetector::readParameters() {
  // Load common parameters.
  nodeHandle_.param("image_view/enable_opencv", viewImage_, true);
  nodeHandle_.param("image_view/wait_key_delay", waitKeyDelay_, 3);
  nodeHandle_.param("image_view/enable_console_output", enableConsoleOutput_, false);

  // Check if Xserver is running on Linux.
  if (XOpenDisplay(NULL)) {
    // Do nothing!
    ROS_INFO("[YoloObjectDetector] Xserver is running.");
	viewImage_ = false;
  } else {
    ROS_INFO("[YoloObjectDetector] Xserver is not running.");
    viewImage_ = false;
  }

  // Set vector sizes.
  nodeHandle_.param("yolo_model/detection_classes/names", classLabels_, std::vector<std::string>(0));
  numClasses_ = classLabels_.size();
  rosBoxes_ = std::vector<std::vector<RosBox_> >(numClasses_);
  rosBoxCounter_ = std::vector<int>(numClasses_);

  return true;
}

void YoloObjectDetector::init() {
  ROS_INFO("[YoloObjectDetector] init()."); 

  // Initialize deep network of darknet.
  std::string weightsPath;
  std::string configPath;
  std::string dataPath;
  std::string configModel;
  std::string weightsModel;

  // Threshold of object detection.
  float thresh;
  nodeHandle_.param("yolo_model/threshold/value", thresh, (float)0.5);

  // Path to weights file.
  nodeHandle_.param("yolo_model/weight_file/name", weightsModel, std::string("yolov2-tiny.weights"));
  nodeHandle_.param("weights_path", weightsPath, std::string("/default"));
  weightsPath += "/" + weightsModel;
  weights = new char[weightsPath.length() + 1];
  strcpy(weights, weightsPath.c_str());

  // Path to config file.
  nodeHandle_.param("yolo_model/config_file/name", configModel, std::string("yolov2-tiny.cfg"));
  nodeHandle_.param("config_path", configPath, std::string("/default"));
  configPath += "/" + configModel;
  cfg = new char[configPath.length() + 1];
  strcpy(cfg, configPath.c_str());

  // Path to data folder.
  dataPath = darknetFilePath_;
  dataPath += "/data";
  data = new char[dataPath.length() + 1];
  strcpy(data, dataPath.c_str());

  // Get classes.
  detectionNames = (char**)realloc((void*)detectionNames, (numClasses_ + 1) * sizeof(char*));
  for (int i = 0; i < numClasses_; i++) {
    detectionNames[i] = new char[classLabels_[i].length() + 1];
    strcpy(detectionNames[i], classLabels_[i].c_str());
  }

  // Load network.
  setupNetwork(cfg, weights, data, thresh, detectionNames, numClasses_, 0, 0, 1, 0.5, 0, 0, 0, 0);
  yoloThread_ = std::thread(&YoloObjectDetector::yolo, this);

  // Initialize publisher and subscriber.
  std::string cameraTopicName;
  int cameraQueueSize;
  std::string objectDetectorTopicName;
  int objectDetectorQueueSize;
  bool objectDetectorLatch;
  std::string boundingBoxesTopicName;
  int boundingBoxesQueueSize;
  bool boundingBoxesLatch;
  std::string detectionImageTopicName;
  int detectionImageQueueSize;
  bool detectionImageLatch;

  // publisher for sign_array_msg
  std::string objTopicName;
  int objQueueSize;
  bool objLatch;

  nodeHandle_.param("subscribers/camera_reading/topic", cameraTopicName, std::string("/camera/image_raw"));
  nodeHandle_.param("subscribers/camera_reading/queue_size", cameraQueueSize, 1);
  nodeHandle_.param("publishers/object_detector/topic", objectDetectorTopicName, std::string("found_object"));
  nodeHandle_.param("publishers/object_detector/queue_size", objectDetectorQueueSize, 1);
  nodeHandle_.param("publishers/object_detector/latch", objectDetectorLatch, false);
  nodeHandle_.param("publishers/bounding_boxes/topic", boundingBoxesTopicName, std::string("bounding_boxes"));
  nodeHandle_.param("publishers/bounding_boxes/queue_size", boundingBoxesQueueSize, 1);
  nodeHandle_.param("publishers/bounding_boxes/latch", boundingBoxesLatch, false);
  nodeHandle_.param("publishers/detection_image/topic", detectionImageTopicName, std::string("detection_image"));
  nodeHandle_.param("publishers/detection_image/queue_size", detectionImageQueueSize, 1);
  nodeHandle_.param("publishers/detection_image/latch", detectionImageLatch, true);

  nodeHandle_.param("publishers/obj/topic", objTopicName, std::string("objects"));
  nodeHandle_.param("publishers/obj/queue_size", objQueueSize, 1);
  nodeHandle_.param("publishers/obj/latch", objLatch, false);

  imageSubscriber_ = imageTransport_.subscribe(cameraTopicName, cameraQueueSize, &YoloObjectDetector::cameraCallback, this);
  objectPublisher_ =
      nodeHandle_.advertise<darknet_ros_msgs::ObjectCount>(objectDetectorTopicName, objectDetectorQueueSize, objectDetectorLatch);
  boundingBoxesPublisher_ =
      nodeHandle_.advertise<darknet_ros_msgs::BoundingBoxes>(boundingBoxesTopicName, boundingBoxesQueueSize, boundingBoxesLatch);
  detectionImagePublisher_ =
      nodeHandle_.advertise<sensor_msgs::Image>(detectionImageTopicName, detectionImageQueueSize, detectionImageLatch);

  object_pub =
      nodeHandle_.advertise<vision_msgs::sign_array_msg>(objTopicName, objQueueSize, objLatch);

  // Action servers.
  std::string checkForObjectsActionName;
  nodeHandle_.param("actions/camera_reading/topic", checkForObjectsActionName, std::string("check_for_objects"));
  checkForObjectsActionServer_.reset(new CheckForObjectsActionServer(nodeHandle_, checkForObjectsActionName, false));
  checkForObjectsActionServer_->registerGoalCallback(boost::bind(&YoloObjectDetector::checkForObjectsActionGoalCB, this));
  checkForObjectsActionServer_->registerPreemptCallback(boost::bind(&YoloObjectDetector::checkForObjectsActionPreemptCB, this));
  checkForObjectsActionServer_->start();
}

void YoloObjectDetector::cameraCallback(const sensor_msgs::ImageConstPtr& msg) {
  ROS_DEBUG("[YoloObjectDetector] USB image received.");

  cv_bridge::CvImagePtr cam_image;

  try {
    cam_image = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
    cam_image->image(cv::Rect(832, 416, 832, 624)).copyTo(cam_image->image);
  } catch (cv_bridge::Exception& e) {
    ROS_ERROR("cv_bridge exception: %s", e.what());
    return;
  }

  if (cam_image) {
    {
      boost::unique_lock<boost::shared_mutex> lockImageCallback(mutexImageCallback_);
      imageHeader_ = msg->header;
      camImageCopy_ = cam_image->image.clone();
    }
    {
      boost::unique_lock<boost::shared_mutex> lockImageStatus(mutexImageStatus_);
      imageStatus_ = true;
    }
    frameWidth_ = cam_image->image.size().width;
    frameHeight_ = cam_image->image.size().height;
  }
  return;
}

void YoloObjectDetector::checkForObjectsActionGoalCB() {
  ROS_DEBUG("[YoloObjectDetector] Start check for objects action.");

  boost::shared_ptr<const darknet_ros_msgs::CheckForObjectsGoal> imageActionPtr = checkForObjectsActionServer_->acceptNewGoal();
  sensor_msgs::Image imageAction = imageActionPtr->image;

  cv_bridge::CvImagePtr cam_image;

  try {
    cam_image = cv_bridge::toCvCopy(imageAction, sensor_msgs::image_encodings::BGR8);
  } catch (cv_bridge::Exception& e) {
    ROS_ERROR("cv_bridge exception: %s", e.what());
    return;
  }

  if (cam_image) {
    {
      boost::unique_lock<boost::shared_mutex> lockImageCallback(mutexImageCallback_);
      camImageCopy_ = cam_image->image.clone();
    }
    {
      boost::unique_lock<boost::shared_mutex> lockImageCallback(mutexActionStatus_);
      actionId_ = imageActionPtr->id;
    }
    {
      boost::unique_lock<boost::shared_mutex> lockImageStatus(mutexImageStatus_);
      imageStatus_ = true;
    }
    frameWidth_ = cam_image->image.size().width;
    frameHeight_ = cam_image->image.size().height;
  }
  return;
}

void YoloObjectDetector::checkForObjectsActionPreemptCB() {
  ROS_DEBUG("[YoloObjectDetector] Preempt check for objects action.");
  checkForObjectsActionServer_->setPreempted();
}

bool YoloObjectDetector::isCheckingForObjects() const {
  return (ros::ok() && checkForObjectsActionServer_->isActive() && !checkForObjectsActionServer_->isPreemptRequested());
}

bool YoloObjectDetector::publishDetectionImage(const cv::Mat& detectionImage) {
  if (detectionImagePublisher_.getNumSubscribers() < 1) return false;
  cv_bridge::CvImage cvImage;
  cvImage.header.stamp = ros::Time::now();
  cvImage.header.frame_id = "detection_image";
  cvImage.encoding = sensor_msgs::image_encodings::BGR8;
  cvImage.image = detectionImage;
  detectionImagePublisher_.publish(*cvImage.toImageMsg());
  ROS_DEBUG("Detection image has been published.");
  return true;
}

// double YoloObjectDetector::getWallTime()
// {
//   struct timeval time;
//   if (gettimeofday(&time, NULL)) {
//     return 0;
//   }
//   return (double) time.tv_sec + (double) time.tv_usec * .000001;
// }

int YoloObjectDetector::sizeNetwork(network* net) {
  int i;
  int count = 0;
  for (i = 0; i < net->n; ++i) {
    layer l = net->layers[i];
    if (l.type == YOLO || l.type == REGION || l.type == DETECTION) {
      count += l.outputs;
    }
  }
  return count;
}

void YoloObjectDetector::rememberNetwork(network* net) {
  int i;
  int count = 0;
  for (i = 0; i < net->n; ++i) {
    layer l = net->layers[i];
    if (l.type == YOLO || l.type == REGION || l.type == DETECTION) {
      memcpy(predictions_[demoIndex_] + count, net->layers[i].output, sizeof(float) * l.outputs);
      count += l.outputs;
    }
  }
}

detection* YoloObjectDetector::avgPredictions(network* net, int* nboxes) {
  int i, j;
  int count = 0;
  fill_cpu(demoTotal_, 0, avg_, 1);
  for (j = 0; j < demoFrame_; ++j) {
    axpy_cpu(demoTotal_, 1. / demoFrame_, predictions_[j], 1, avg_, 1);
  }
  for (i = 0; i < net->n; ++i) {
    layer l = net->layers[i];
    if (l.type == YOLO || l.type == REGION || l.type == DETECTION) {
      memcpy(l.output, avg_ + count, sizeof(float) * l.outputs);
      count += l.outputs;
    }
  }
  // detection* dets = get_network_boxes(net, buff_[0].w, buff_[0].h, demoThresh_, demoHier_, 0, 1, nboxes);
  detection* dets = get_network_boxes(net, buff_[0].w, buff_[0].h, demoThresh_, demoHier_, 0, 1, nboxes, 1);
  return dets;
}

void* YoloObjectDetector::detectInThread() {
  running_ = 1;
  float nms = .4;

  layer l = net_->layers[net_->n - 1];
  float* X = buffLetter_[(buffIndex_ + 2) % 3].data;
  float* prediction = network_predict(*net_, X);

  rememberNetwork(net_);
  detection* dets = 0;
  int nboxes = 0;
  dets = avgPredictions(net_, &nboxes);

  if (nms > 0) do_nms_obj(dets, nboxes, l.classes, nms);

  if (enableConsoleOutput_) {
    //printf("\033[2J");
    //printf("\033[1;1H");
    //printf("\nFPS:%.1f\n", fps_);
    //printf("\nFPS2:%.1f\n", fps2_);
    //printf("Objects:\n\n");
  }
  image display = buff_[(buffIndex_ + 2) % 3];
  // draw_detections(display, dets, nboxes, demoThresh_, demoNames_, demoAlphabet_, demoClasses_, 1);
  draw_detections_v3(display, dets, nboxes, demoThresh_, demoNames_, demoAlphabet_, demoClasses_, 1);


  // extract the bounding boxes and send them to ROS
  int i, j;
  int count = 0;
  for (i = 0; i < nboxes; ++i) {
    float xmin = dets[i].bbox.x - dets[i].bbox.w / 2.;
    float xmax = dets[i].bbox.x + dets[i].bbox.w / 2.;
    float ymin = dets[i].bbox.y - dets[i].bbox.h / 2.;
    float ymax = dets[i].bbox.y + dets[i].bbox.h / 2.;

    if (xmin < 0) xmin = 0;
    if (ymin < 0) ymin = 0;
    if (xmax > 1) xmax = 1;
    if (ymax > 1) ymax = 1;

    // iterate through possible boxes and collect the bounding boxes
    for (j = 0; j < demoClasses_; ++j) {
      if (dets[i].prob[j]) {
        float x_center = (xmin + xmax) / 2;
        float y_center = (ymin + ymax) / 2;
        float BoundingBox_width = xmax - xmin;
        float BoundingBox_height = ymax - ymin;

        // define bounding box
        // BoundingBox must be 1% size of frame (3.2x2.4 pixels)
        if (BoundingBox_width > 0.01 && BoundingBox_height > 0.01) {
          roiBoxes_[count].x = x_center;
          roiBoxes_[count].y = y_center;
          roiBoxes_[count].w = BoundingBox_width;
          roiBoxes_[count].h = BoundingBox_height;
          roiBoxes_[count].Class = j;
          roiBoxes_[count].prob = dets[i].prob[j];
          count++;
        }
      }
    }
  }

  // create array to store found bounding boxes
  // if no object detected, make sure that ROS knows that num = 0
  if (count == 0) {
    roiBoxes_[0].num = 0;
  } else {
    roiBoxes_[0].num = count;
  }

  free_detections(dets, nboxes);
  demoIndex_ = (demoIndex_ + 1) % demoFrame_;
  running_ = 0;
  return 0;
}

void* YoloObjectDetector::fetchInThread() {
  {
    boost::shared_lock<boost::shared_mutex> lock(mutexImageCallback_);
    IplImageWithHeader_ imageAndHeader = getIplImageWithHeader();
    IplImage* ROS_img = imageAndHeader.image;
    ipl_into_image(ROS_img, buff_[buffIndex_]);
    headerBuff_[buffIndex_] = imageAndHeader.header;
    buffId_[buffIndex_] = actionId_;
  }
  rgbgr_image(buff_[buffIndex_]);
  letterbox_image_into(buff_[buffIndex_], net_->w, net_->h, buffLetter_[buffIndex_]);
  return 0;
}

void* YoloObjectDetector::displayInThread(void* ptr) {
  show_image_cv(buff_[(buffIndex_ + 1) % 3], "YOLO V4");
  int c = cv::waitKey(waitKeyDelay_);
  if (c != -1) c = c % 256;
  if (c == 27) {
    demoDone_ = 1;
    return 0;
  } else if (c == 82) {
    demoThresh_ += .02;
  } else if (c == 84) {
    demoThresh_ -= .02;
    if (demoThresh_ <= .02) demoThresh_ = .02;
  } else if (c == 83) {
    demoHier_ += .02;
  } else if (c == 81) {
    demoHier_ -= .02;
    if (demoHier_ <= .0) demoHier_ = .0;
  }
  return 0;
}

void* YoloObjectDetector::displayLoop(void* ptr) {
  while (1) {
    displayInThread(0);
  }
}

void* YoloObjectDetector::detectLoop(void* ptr) {
  while (1) {
    detectInThread();
  }
}

void YoloObjectDetector::setupNetwork(char* cfgfile, char* weightfile, char* datafile, float thresh, char** names, int classes, int delay,
                                      char* prefix, int avg_frames, float hier, int w, int h, int frames, int fullscreen) {
  demoPrefix_ = prefix;
  demoDelay_ = delay;
  demoFrame_ = avg_frames;
  image** alphabet = load_alphabet_with_file(datafile);
  demoNames_ = names;
  demoAlphabet_ = alphabet;
  demoClasses_ = classes;
  demoThresh_ = thresh;
  demoHier_ = hier;
  fullScreen_ = fullscreen;
  printf("YOLO V4\n");
  net_ = load_network(cfgfile, weightfile, 0);
  set_batch_network(net_, 1);
}

void YoloObjectDetector::yolo() {
  const auto wait_duration = std::chrono::milliseconds(2000);
  while (!getImageStatus()) {
    //printf("Waiting for image.\n");
    if (!isNodeRunning()) {
      return;
    }
    std::this_thread::sleep_for(wait_duration);
  }

  std::thread detect_thread;
  std::thread fetch_thread;

  srand(2222222);

  int i;
  demoTotal_ = sizeNetwork(net_);
  predictions_ = (float**)calloc(demoFrame_, sizeof(float*));
  for (i = 0; i < demoFrame_; ++i) {
    predictions_[i] = (float*)calloc(demoTotal_, sizeof(float));
  }
  avg_ = (float*)calloc(demoTotal_, sizeof(float));

  layer l = net_->layers[net_->n - 1];
  roiBoxes_ = (darknet_ros::RosBox_*)calloc(l.w * l.h * l.n, sizeof(darknet_ros::RosBox_));

  {
    boost::shared_lock<boost::shared_mutex> lock(mutexImageCallback_);
    IplImageWithHeader_ imageAndHeader = getIplImageWithHeader();
    IplImage* ROS_img = imageAndHeader.image;
    buff_[0] = ipl_to_image(ROS_img);
    headerBuff_[0] = imageAndHeader.header;
  }
  buff_[1] = copy_image(buff_[0]);
  buff_[2] = copy_image(buff_[0]);
  headerBuff_[1] = headerBuff_[0];
  headerBuff_[2] = headerBuff_[0];
  buffLetter_[0] = letterbox_image(buff_[0], net_->w, net_->h);
  buffLetter_[1] = letterbox_image(buff_[0], net_->w, net_->h);
  buffLetter_[2] = letterbox_image(buff_[0], net_->w, net_->h);
  ipl_ = cvCreateImage(cvSize(buff_[0].w, buff_[0].h), IPL_DEPTH_8U, buff_[0].c);

  int count = 0;

  if (!demoPrefix_ && viewImage_) {
    cv::namedWindow("YOLO V4", cv::WINDOW_NORMAL);
    if (fullScreen_) {
      cv::setWindowProperty("YOLO V4", cv::WND_PROP_FULLSCREEN, cv::WINDOW_FULLSCREEN);
    } else {
      cv::moveWindow("YOLO V4", 0, 0);
      cv::resizeWindow("YOLO V4", 640, 480);
    }
  }

  demoTime_ = what_time_is_it_now();
//added
demoTime2_ = what_time_is_it_now();

  while (!demoDone_) {
    time0 = ros::Time::now();
    buffIndex_ = (buffIndex_ + 1) % 3;
    fetch_thread = std::thread(&YoloObjectDetector::fetchInThread, this);
    detect_thread = std::thread(&YoloObjectDetector::detectInThread, this);
    if (!demoPrefix_) {
      fps_ = 1. / (what_time_is_it_now() - demoTime_);
      demoTime_ = what_time_is_it_now();
      if (viewImage_) {
        displayInThread(0);
      } else {
        //generate_image(buff_[(buffIndex_ + 1) % 3], ipl_);
      }
      publishInThread();
//added
fps2_ = 1. / (what_time_is_it_now() - demoTime2_);
//added
demoTime2_ = what_time_is_it_now();
    } else {
      char name[256];
      sprintf(name, "%s_%08d", demoPrefix_, count);
      save_image(buff_[(buffIndex_ + 1) % 3], name);
    }
    fetch_thread.join();
    detect_thread.join();
    ++count;
    if (!isNodeRunning()) {
      demoDone_ = true;
    }
  }
}

IplImageWithHeader_ YoloObjectDetector::getIplImageWithHeader() {
  IplImage* ROS_img = new IplImage(camImageCopy_);
  IplImageWithHeader_ header = {.image = ROS_img, .header = imageHeader_};
  return header;
}

bool YoloObjectDetector::getImageStatus(void) {
  boost::shared_lock<boost::shared_mutex> lock(mutexImageStatus_);
  return imageStatus_;
}

bool YoloObjectDetector::isNodeRunning(void) {
  boost::shared_lock<boost::shared_mutex> lock(mutexNodeStatus_);
  return isNodeRunning_;
}

void* YoloObjectDetector::publishInThread() {
  // Publish image.
  cv::Mat cvImage = cv::cvarrToMat(ipl_);

  // save image as Mat 
  cv::Mat cur_image = image_to_mat(buff_[(buffIndex_ + 1) % 3]);
  //cv::imshow("m", m);
  //cv::waitKey(0);

  if (!publishDetectionImage(cv::Mat(cvImage))) {
    ROS_DEBUG("Detection image has not been broadcasted.");
  }

  // Publish bounding boxes and detection result.
  int num = roiBoxes_[0].num;
  if (num > 0 && num <= 100) {
    for (int i = 0; i < num; i++) {
      for (int j = 0; j < numClasses_; j++) {
        if (roiBoxes_[i].Class == j) {
          rosBoxes_[j].push_back(roiBoxes_[i]);
          rosBoxCounter_[j]++;
        }
      }
    }

    darknet_ros_msgs::ObjectCount msg;
    msg.header.stamp = time0;
    msg.header.frame_id = "detection";
    msg.count = num;
    objectPublisher_.publish(msg);

    bboxs.header.stamp = imageHeader_.stamp;
    for (int i = 0; i < numClasses_; i++) {
      if (rosBoxCounter_[i] > 0) {
        darknet_ros_msgs::BoundingBox boundingBox;

        vision_msgs::sign_detection_msg sign_detection;

        for (int j = 0; j < rosBoxCounter_[i]; j++) {
          int xmin = (rosBoxes_[i][j].x - rosBoxes_[i][j].w / 2) * frameWidth_;
          int ymin = (rosBoxes_[i][j].y - rosBoxes_[i][j].h / 2) * frameHeight_;
          int xmax = (rosBoxes_[i][j].x + rosBoxes_[i][j].w / 2) * frameWidth_;
          int ymax = (rosBoxes_[i][j].y + rosBoxes_[i][j].h / 2) * frameHeight_;

          boundingBox.Class = classLabels_[i];
          boundingBox.id = i;
          boundingBox.probability = rosBoxes_[i][j].prob;
          boundingBox.xmin = xmin;
          boundingBox.ymin = ymin;
          boundingBox.xmax = xmax;
          boundingBox.ymax = ymax;
          boundingBoxesResults_.bounding_boxes.push_back(boundingBox);
	  
          // find distance from front of vehicle: (focal(mm) * real height(mm) * image height(pixels)) / (object height(pixels) * sensor height(mm))
	      
          // convert focal length found via camera matrix from pixels to mm
     /*     
	double focal = 1508.986 * .005498;	

          double numerator;
          double denominator;
          if (classLabels_[i] == "speed limit" || classLabels_[i] == "stop")    // both same height 
            numerator = focal * 762 * 2048;
            denominator = (ymax - ymin) * 11.26;
            sign_detection.z = (numerator / denominator) / 1000;	
	      if (classLabels_[i] == "pedestrian crossing") 	
            numerator = focal * 609.6 * 2048;
            denominator = (ymax - ymin) * 11.26;
            sign_detection.z = (numerator / denominator) / 1000;
	      if (classLabels_[i] == "yield") 
            numerator = focal * 914.4 * 2048;
            denominator = (ymax - ymin) * 11.26;
            sign_detection.z = (numerator / denominator) / 1000;
          */

	double fx = 442.78;
	double fy = 442.78;
	double perceived_depth_x = ((59973*fx)/ (xmax-xmin) ) * 0.00001877934;
	double perceived_depth_y = ((59973*fx)/ (ymax-ymin) ) * 0.00001877934;
	double estimated_distance = (perceived_depth_x + perceived_depth_y)/2;
	sign_detection.z = estimated_distance;
          // initialize values in sign_detection_msg 
          sign_detection.header = bboxs.header;
          sign_detection.coor_x = xmin;
          sign_detection.coor_y = ymax;
          sign_detection.size_x = xmax - xmin;
          sign_detection.size_y = ymax - ymin;

          // detect number if classified sign is speed limit
	      if (classLabels_[i] == "speed limit") {
            // create Mat of just the speed limit sign
            cv::Rect crop_region(xmin-3, ymin-3, round(1.1*(xmax - xmin)), round(1.1*(ymax - ymin)));
	        cv::Mat cropped;
		cur_image(crop_region).copyTo(cropped);
	        // std::cout << image2 << std::endl;
	        // cv::imshow("image2", image2);
            // cv::waitKey(0);
        
            // convert to grayscale
	        cv::Mat graymat;
	        cv::cvtColor(cropped, graymat, CV_BGR2GRAY);

            // resize image
	        cv::Size size(60, 75);
	        cv::resize(graymat, graymat, size);

            // convert to binary 
	        cv::threshold(graymat, graymat, 70, 255, cv::THRESH_BINARY);

            // find contours 
	        std::vector<std::vector<cv::Point> > contours;
	        findContours(graymat, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
	        // std::cout << contours[0];
	        cv::Rect r = cv::boundingRect(contours[0]);
	        // auto width = r.width;
	        // auto height = r.height;
	        // auto x = r.x;
	        // auto y = r.y;
	        // std::cout << width << " " << height << " " << x << " " << y << std::endl;
  	        
            // invert image
	        cv::Mat thresh = graymat(r);
	        cv::bitwise_not(thresh, thresh);

            // find contours again
	        std::vector<std::vector<cv::Point> > contours2;
	        findContours(thresh, contours2, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
	        // std::cout << contours2.size();
	  
            // crop left and right digit separately
	        int left_x = 60;
	        int count = 0;
	        cv::Mat left;
	        cv::Mat right;
	        cv::Mat roi;

	        for (size_t i = 0; i < contours2.size(); i++) {
	  	      cv::Rect rec = cv::boundingRect(contours2[i]);
		      roi = thresh(rec);
		      if (rec.width > 10 && rec.height > 10 && rec.height < 40 && rec.width < 35 && (rec.width / rec.height) < 2 &&
			     (rec.height / rec.width) < 2 && rec.x != 0 && rec.y != 0 && rec.x + rec.width != 60 && rec.y + rec.height != 75) {
			    count++;
		        if (count == 1) {
			      left = roi;
			      right = roi;
		        }
		        if (rec.x < left_x) {
			      left = roi;
			      left_x = rec.x;
		        } else {
			      right = roi;
		        }
		      }
	        }

	        if (count != 2) {
			break;		
		}
            // add black background behind number
	        cv::Mat blackLeft(28, 28, CV_8UC1, cv::Scalar(0));
	        cv::Size sizeLeft(22, 22);
	        cv::resize(left, left, sizeLeft);
	        left.copyTo(blackLeft(cv::Rect(3, 3, left.cols, left.rows)));
	        // cv::imshow("black", blackLeft);
	        // cv::waitKey(0);

	        cv::Mat blackRight(28, 28, CV_8UC1, cv::Scalar(0));
	        cv::Size sizeRight(22, 22);
	        cv::resize(right, right, sizeRight);
	        right.copyTo(blackRight(cv::Rect(3, 3, right.cols, right.rows)));
	        // cv::imshow("black", blackRight);
	        // cv::waitKey(0);
	
	        // convert Mat to 2D vector
            cv::Mat dstLeft;
	        cv::Mat dstRight;

	        // Convert to double (much faster than a simple for loop)
	        blackLeft.convertTo(dstLeft, CV_64F, 1, 0);
	        double *ptrDstLeft[dstLeft.rows];

	
	        std::vector<std::vector<float>> vec;
	        for (int i = 0; i<28; i ++) {
		        std::vector<float> c;
		        ptrDstLeft[i] = dstLeft.ptr<double>(i);
		        for (int j = 0; j<28; j++) {
			        double value = ptrDstLeft[i][j] / 255.0;
			        c.push_back(value);
			        // std::cout << value << " ";
		        }
		        // std::cout << std::endl;
		        vec.push_back(c);
	        }
	

	        blackRight.convertTo(dstRight, CV_64F, 1, 0);
	        double *ptrDstRight[dstRight.rows];
	
	        std::vector<std::vector<float>> vec2;
	        for (int i = 0; i<28; i ++) {
		        std::vector<float> c;
		        ptrDstRight[i] = dstRight.ptr<double>(i);
		        for (int j = 0; j<28; j++) {
			        double value = ptrDstRight[i][j] / 255.0;
			        c.push_back(value);
			        // std::cout << value << " ";
		        }
		        // std::cout << std::endl;
		        vec2.push_back(c);
	        }
	//std::cout << "print out plsssss before CNN";
            // input 2D vector into CNN model and get results
  	        std::vector<float> f;
	        std::vector<float> f2;
	
	        // std::cout << black << std::endl;
	        // cv::imshow("left", left);
	        // cv::waitKey(0);
	        // cv::imshow("right", right);
	        // cv::waitKey(0);
	        // cv::imshow("thresh", thresh);
	
	        DataChunk *sample = new DataChunk2D();
	        sample->read_vector(vec);
	        //std::cout << "sample 3d size: " << sample->get_3d().size() << std::endl;
	        
	        f = km->compute_output(sample);

	        //std::cout << "DataChunkFlat values (Left Digit):" << std::endl;
	        int digit = -1;
	        float percent = 0.0;
	        for(size_t i = 0; i < f.size(); ++i) {
		      //std::cout << f[i] << " ";
	  	      //std::cout << std::endl;
		      if (f[i] > percent) {
			    digit = i;
			    percent = f[i];
		      }
	        }

	        DataChunk *sample2 = new DataChunk2D();
	        sample2->read_vector(vec2);

	        f2 = km->compute_output(sample2);

	        //std::cout << "DataChunkFlat values (Right Digit):" << std::endl;
	        int digit2 = -1;
	        float percent2 = 0.0;
	        for(size_t i = 0; i < f2.size(); ++i) {
		      //std::cout << f2[i] << " ";
	  	      //std::cout << std::endl;
		      if (f2[i] > percent2) {
			    digit2 = i;
			    percent2 = f2[i];
		      }
	        }

	        delete sample;
	        delete sample2;
	        
            // update sign_detection_msg what variable with speed limit + number
	        if (percent >= .60) {
		        sign_detection.what = classLabels_[i] + " " + std::to_string(digit) + std::to_string(digit2);
	        } else {
		        sign_detection.what = classLabels_[i];
	        }
 	      } else {
		    sign_detection.what = classLabels_[i];
	      }
	      bboxs.s.push_back(sign_detection);
        }
      }
    }

    boundingBoxesResults_.header.stamp = ros::Time::now();
    boundingBoxesResults_.header.frame_id = "detection";
    boundingBoxesResults_.image_header = headerBuff_[(buffIndex_ + 1) % 3];
    boundingBoxesPublisher_.publish(boundingBoxesResults_);

    object_pub.publish(bboxs);
  } else {
    darknet_ros_msgs::ObjectCount msg;
    msg.header.stamp = ros::Time::now();
    msg.header.frame_id = "detection";
    msg.count = 0;
    objectPublisher_.publish(msg);
  }
  if (isCheckingForObjects()) {
    ROS_DEBUG("[YoloObjectDetector] check for objects in image.");
    darknet_ros_msgs::CheckForObjectsResult objectsActionResult;
    objectsActionResult.id = buffId_[0];
    objectsActionResult.bounding_boxes = boundingBoxesResults_;
    checkForObjectsActionServer_->setSucceeded(objectsActionResult, "Send bounding boxes.");
  }
  boundingBoxesResults_.bounding_boxes.clear();

  bboxs.s.clear();

  for (int i = 0; i < numClasses_; i++) {
    rosBoxes_[i].clear();
    rosBoxCounter_[i] = 0;
  }

  return 0;
}

} /* namespace darknet_ros*/



