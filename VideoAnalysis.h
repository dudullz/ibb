
#pragma once

#include <iostream>
#include <sstream>

#include "VideoCapture.h"
#include "FrameProcessor.h"

namespace ibb
{
  class VideoAnalysis
  {
  private:
    VideoCapture* videoCapture;
    FrameProcessor* frameProcessor;
    bool use_file;
    std::string filename;
    bool use_camera;
    int cameraIndex;
    bool use_comp;
    long frameToStop;
    std::string imgref;
		bool use_ubd;
	
	bool use_imgs;
	std::string imgPath;
	std::string savePath;
	std::string probPath;
	std::string csvPath;
	std::string vaUBModel;
	std::string vaFaceModel;

  public:
    VideoAnalysis();
    ~VideoAnalysis();

    bool setup(int argc, const char **argv);
    void start();
  };
}
