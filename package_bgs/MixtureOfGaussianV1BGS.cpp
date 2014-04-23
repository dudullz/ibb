
#include "MixtureOfGaussianV1BGS.h"

MixtureOfGaussianV1BGS::MixtureOfGaussianV1BGS() : firstTime(true), alpha(0.05), enableThreshold(true), threshold(15), showOutput(true)
{
  std::cout << "MixtureOfGaussianV1BGS()" << std::endl;
}

MixtureOfGaussianV1BGS::~MixtureOfGaussianV1BGS()
{
  std::cout << "~MixtureOfGaussianV1BGS()" << std::endl;
}

void MixtureOfGaussianV1BGS::process(const cv::Mat &img_input, cv::Mat &img_output, cv::Mat &img_bgmodel)
{
  if(img_input.empty())
    return;

  loadConfig();

  if(firstTime)
    saveConfig();

  //------------------------------------------------------------------
  // BackgroundSubtractorMOG
  // http://opencv.itseez.com/modules/video/doc/motion_analysis_and_object_tracking.html#backgroundsubtractormog
  //
  // Gaussian Mixture-based Backbround/Foreground Segmentation Algorithm.
  //
  // The class implements the algorithm described in:
  //   P. KadewTraKuPong and R. Bowden, 
  //   An improved adaptive background mixture model for real-time tracking with shadow detection, 
  //   Proc. 2nd European Workshp on Advanced Video-Based Surveillance Systems, 2001
  //------------------------------------------------------------------

  mog(img_input, img_foreground, alpha);
  cv::Mat img_background;
  mog.getBackgroundImage(img_background);

  if(enableThreshold)
    cv::threshold(img_foreground, img_foreground, threshold, 255, cv::THRESH_BINARY);

  if(showOutput)
  {
    if (!img_foreground.empty())
      cv::imshow("GMM FG (KadewTraKuPong&Bowden)", img_foreground);
    
    if (!img_background.empty())
      cv::imshow("GMM BG (KadewTraKuPong&Bowden)", img_background);
  }

  img_foreground.copyTo(img_output);
  img_background.copyTo(img_bgmodel);

  firstTime = false;
}

void MixtureOfGaussianV1BGS::saveConfig()
{
#if defined(_WIN32)
	//CvFileStorage* fs = cvOpenFileStorage("F:\\Developer\\BGS\\AndrewsSobral\\bgslibrary\\config\\MixtureOfGaussianV1BGS.xml", 0, CV_STORAGE_WRITE);
	CvFileStorage* fs = cvOpenFileStorage("config\\MixtureOfGaussianV1BGS.xml", 0, CV_STORAGE_WRITE);
#else
    CvFileStorage* fs = cvOpenFileStorage("./config/MixtureOfGaussianV1BGS.xml", 0, CV_STORAGE_WRITE);
#endif
  
  cvWriteReal(fs, "alpha", alpha);
  cvWriteInt(fs, "enableThreshold", enableThreshold);
  cvWriteInt(fs, "threshold", threshold);
  cvWriteInt(fs, "showOutput", showOutput);

  cvReleaseFileStorage(&fs);
}

void MixtureOfGaussianV1BGS::loadConfig()
{
#if defined(_WIN32)
	//CvFileStorage* fs = cvOpenFileStorage("F:\\Developer\\BGS\\AndrewsSobral\\bgslibrary\\config\\MixtureOfGaussianV1BGS.xml", 0, CV_STORAGE_READ);
	CvFileStorage* fs = cvOpenFileStorage("config\\MixtureOfGaussianV1BGS.xml", 0, CV_STORAGE_READ);
#else
    CvFileStorage* fs = cvOpenFileStorage("./config/MixtureOfGaussianV1BGS.xml", 0, CV_STORAGE_READ);
#endif
    
  alpha = cvReadRealByName(fs, 0, "alpha", 0.05);
  enableThreshold = cvReadIntByName(fs, 0, "enableThreshold", true);
  threshold = cvReadIntByName(fs, 0, "threshold", 15);
  showOutput = cvReadIntByName(fs, 0, "showOutput", true);

  cvReleaseFileStorage(&fs);
}
