
#include "VideoAnalysis.h"

namespace bgslibrary
{
  VideoAnalysis::VideoAnalysis() : use_file(false), use_camera(false), use_imgs(false), cameraIndex(0), use_comp(false), frameToStop(0)
  {
    std::cout << "VideoAnalysis()" << std::endl;
  }

  VideoAnalysis::~VideoAnalysis()
  {
    std::cout << "~VideoAnalysis()" << std::endl;
  }

  bool VideoAnalysis::setup(int argc, const char **argv)
  {
    bool flag = false;

    const char* keys =
      "{hp|help|false|Print help message}"
      "{uf|use_file|false|Use video file}"
      "{fn|filename||Specify video file}"
      "{uc|use_cam|false|Use camera}"
      "{ca|camera|0|Specify camera index}"
      "{co|use_comp|false|Use mask comparator}"
      "{st|stopAt|0|Frame number to stop}"
      "{im|imgref||Specify image file}"
	  	"{pt|imgpath||Specify the absolute path to the input image sequence}"
		"{sv|save_path||Specify the absolute path to save the output binary images}"
		"{prob|prob_path||Specify the absolute path to save the probability map images, if any}"
		"{csv|csv_path||Specify the absolute path to save the probability map to csv file, if any}"
		"{ub|ub_model||Specify the absolute path to the upper boddy model file}"
		"{face|face_model||Specify the absolute path to the face model file}"
      ;
    cv::CommandLineParser cmd(argc, argv, keys);

    if (argc <= 1 || cmd.get<bool>("help") == true)
    {
      std::cout << "Usage: " << argv[0] << " [options]" << std::endl;
      std::cout << "Avaible options:" << std::endl;
      cmd.printParams();
      return false;
    }

    use_file = cmd.get<bool>("use_file");
    if (use_file)
    {
      filename = cmd.get<std::string>("filename");

      if (filename.empty())
      {
        std::cout << "Specify filename" << std::endl;
        return false;
      }

      flag = true;
    }

    use_camera = cmd.get<bool>("use_cam");
    if (use_camera)
    {
      cameraIndex = cmd.get<int>("camera");
      flag = true;
    }
    
     if(!use_file && !use_camera)
	  use_imgs = true;
// 	use_imgs = cmd.get<bool>("use_imgs");
	if(use_imgs)
	{
		imgPath = cmd.get<std::string>("imgpath");
		std::cout << "	Get images from " << imgPath << std::endl;
		flag = true;
	}
	
	savePath = cmd.get<std::string>("save_path");
	probPath = cmd.get<std::string>("prob_path");
	csvPath = cmd.get<std::string>("csv_path");
	vaUBModel = cmd.get<std::string>("ub_model");
	vaFaceModel = cmd.get<std::string>("face_model");

    if (flag == true)
    {
      use_comp = cmd.get<bool>("use_comp");
      if (use_comp)
      {
        frameToStop = cmd.get<int>("stopAt");
        imgref = cmd.get<std::string>("imgref");

        if (imgref.empty())
        {
          std::cout << "Specify image reference" << std::endl;
          return false;
        }
      }
    }

    return flag;
  }

  void VideoAnalysis::start()
  {
    do
    {
      videoCapture = new VideoCapture;
      frameProcessor = new FrameProcessor;

      frameProcessor->init();
      frameProcessor->frameToStop = frameToStop;
      frameProcessor->imgref = imgref;
	  frameProcessor->savePath = savePath;
	  frameProcessor->probPath = probPath;
	  frameProcessor->csvPath = csvPath;
			
			if(vaUBModel.length() > 0)
				frameProcessor->setUpperBodyDetector(vaUBModel);
			
			if(vaFaceModel.length() > 0)
				frameProcessor->setFaceDetector(vaFaceModel);

      videoCapture->setFrameProcessor(frameProcessor);

      if (use_file)
        videoCapture->setVideo(filename);

      if (use_camera)
        videoCapture->setCamera(cameraIndex);

	  if(use_imgs)
		videoCapture->setImages(imgPath);
	  
      videoCapture->start();

      if (use_file || use_camera)
        break;

      frameProcessor->finish();

      int key = cvWaitKey(500);
      if (key == KEY_ESC)
        break;

      delete frameProcessor;
      delete videoCapture;

    } while (1);

    delete frameProcessor;
    delete videoCapture;
  }
}
