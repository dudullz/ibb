
#pragma once

#include <iostream>
#include <cv.h>
#include <highgui.h>
#include <vector>
#include "boost/filesystem.hpp"

#include "Config.h"
#include "IFrameProcessor.h"

namespace ibb
{
  class VideoCapture
  {
  private:
    IFrameProcessor* frameProcessor;
    CvCapture* capture;
    IplImage* frame;
    int key;
    int64 start_time;
    int64 delta_time;
    double freq;
    double fps;
    long frameNumber; // this is not frame number!!! it's the index number for std::vector imgNames
    long stopAt;
    bool useCamera;
    int cameraIndex;
    bool useVideo;
    std::string videoFileName;
    int input_resize_percent;
    bool showOutput;
    bool enableFlip;

	bool useImages;
	std::string imgPath;
  std::vector<std::string> imgFiles;
	typedef std::vector<boost::filesystem::path> vec;             // store paths,
	vec imgNames;                                // so we can sort them later
	
  public:
    VideoCapture();
    ~VideoCapture();

    void setFrameProcessor(IFrameProcessor* frameProcessorPtr);
    void setCamera(int cameraIndex);
    void setVideo(std::string filename);
	void setImages(std::string path);
    void start();

  private:
    void setUpCamera();
    void setUpVideo();
	void setUpImages();

    void saveConfig();
    void loadConfig();
  };
}
