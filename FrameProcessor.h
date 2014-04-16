/*
This file is part of BGSLibrary.

BGSLibrary is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

BGSLibrary is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with BGSLibrary.  If not, see <http://www.gnu.org/licenses/>.
*/
#pragma once
#pragma warning(disable : 4482)

#include "IFrameProcessor.h"
#include "PreProcessor.h"

#include "package_bgs/IBGS.h"

#include "package_bgs/FrameDifferenceBGS.h"
#include "package_bgs/StaticFrameDifferenceBGS.h"
#include "package_bgs/WeightedMovingMeanBGS.h"
#include "package_bgs/WeightedMovingVarianceBGS.h"
#include "package_bgs/MixtureOfGaussianV1BGS.h"
#include "package_bgs/MixtureOfGaussianV2BGS.h"
#include "package_bgs/AdaptiveBackgroundLearning.h"
#if CV_MAJOR_VERSION >= 2 && CV_MINOR_VERSION >= 4 && CV_SUBMINOR_VERSION >= 3
#include "package_bgs/GMG.h"
#endif

// The PBAS algorithm was removed from BGSLibrary because it is
// based on patented algorithm ViBE
// http://www2.ulg.ac.be/telecom/research/vibe/
#include "package_bgs/pt/PixelBasedAdaptiveSegmenter.h"

namespace ibb
{
  class FrameProcessor : public IFrameProcessor
  {
  private:
    bool firstTime;
    long frameNumber;
    std::string processname;
    double duration;
    std::string tictoc;

    cv::Mat img_prep;
    PreProcessor* preProcessor;
    bool enablePreProcessor;

    cv::Mat img_framediff;
    FrameDifferenceBGS* frameDifference;
    bool enableFrameDifferenceBGS;

    cv::Mat img_staticfdiff;
    StaticFrameDifferenceBGS* staticFrameDifference;
    bool enableStaticFrameDifferenceBGS;

    cv::Mat img_wmovmean;
    WeightedMovingMeanBGS* weightedMovingMean;
    bool enableWeightedMovingMeanBGS;

    cv::Mat img_movvar;
    WeightedMovingVarianceBGS* weightedMovingVariance;
    bool enableWeightedMovingVarianceBGS;

    cv::Mat img_mog1;
    MixtureOfGaussianV1BGS* mixtureOfGaussianV1BGS;
    bool enableMixtureOfGaussianV1BGS;

    cv::Mat img_mog2;
    MixtureOfGaussianV2BGS* mixtureOfGaussianV2BGS;
    bool enableMixtureOfGaussianV2BGS;

    cv::Mat img_bkgl_fgmask;
    AdaptiveBackgroundLearning* adaptiveBackgroundLearning;
    bool enableAdaptiveBackgroundLearning;

#if CV_MAJOR_VERSION >= 2 && CV_MINOR_VERSION >= 4 && CV_SUBMINOR_VERSION >= 3
    cv::Mat img_gmg;
    GMG* gmg;
    bool enableGMG;
#endif

    cv::Mat img_pt_pbas;
	cv::Mat img_pt_pbas_prob;
    PixelBasedAdaptiveSegmenter* pixelBasedAdaptiveSegmenter;
    bool enablePBAS;

		cv::LatentSvmDetector lsvm_detector;
		bool enableUpperBodayDetector;
		float overlapThreshold;
		int numThreads;
		std::vector<cv::Scalar> colors;
		
		cv::CascadeClassifier face_detector;
		bool enableFaceDetector;
		
  public:
    FrameProcessor();
    ~FrameProcessor();

    long frameToStop;
    std::string imgref;
	
	std::string savePath;
	std::string saveName;
	std::string probPath;	// prob. image save path
	std::string probName;	// prob. image file name
	std::string csvPath;	// csv  save path
	std::string csvName;	// csv  file name
	std::string fpUBModel;	// Upper Body Model file name
	std::string fpFaceModel;	// Face Model file name

		void setUpperBodyDetector(std::string ub_model);
		void setFaceDetector(std::string face_model);
    void init();
    void process(const cv::Mat &img_input);
    void finish(void);
	void writeProbToCSV(cv::Mat &prob_map);

  private:
    void process(std::string name, IBGS *bgs, const cv::Mat &img_input, cv::Mat &img_bgs);
    void tic(std::string value);
    void toc();

    void saveConfig();
    void loadConfig();
  };
}
