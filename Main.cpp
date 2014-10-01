
#include "Config.h"
#include "VideoAnalysis.h"
#include <iostream>

namespace ibb
{
  class Main
  {
  private:
    Main();

  public:
    static void start(int argc, const char **argv)
    {
      std::cout << "-----------------------------------------" << std::endl;
      std::cout << "Intelligent Billboard Demo V1.0     " << std::endl;
      std::cout << "-----------------------------------------" << std::endl;
      std::cout << "Using OpenCV version " << CV_VERSION << std::endl;

      try
      {
        int key = KEY_ESC;

        do
        {
          VideoAnalysis* videoAnalysis = new VideoAnalysis;

          if (videoAnalysis->setup(argc, argv))
          {
            videoAnalysis->start();

            std::cout << "Processing finished, enter:" << std::endl;
            std::cout << "R - Repeat" << std::endl;
            std::cout << "Q - Quit" << std::endl;

            key = cv::waitKey();
          }

          cv::destroyAllWindows();
          delete videoAnalysis;

        } while (key == KEY_REPEAT);
      }
      catch (const std::exception& ex)
      {
        std::cout << "std::exception:" << ex.what() << std::endl;
        return;
      }
      catch (...)
      {
        std::cout << "Unknow error" << std::endl;
        return;
      }

#ifdef WIN32
      //system("pause");
#endif
    }
  };
}

int main(int argc, const char **argv)
{
  ibb::Main::start(argc, argv);
  return 0;
}
