
#pragma once

#include <cv.h>

namespace ibb
{
  class IFrameProcessor
  {
  public:
    virtual void process(const cv::Mat &input) = 0;
    virtual ~IFrameProcessor(){}
  };
}
