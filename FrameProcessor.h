
#pragma once
#pragma warning(disable : 4482)

#include "IFrameProcessor.h"
#include "PreProcessor.h"

#include "package_bgs/IBGS.h"

#include "package_bgs/WeightedMovingMeanBGS.h"
#include "package_bgs/WeightedMovingVarianceBGS.h"
#include "package_bgs/MixtureOfGaussianV1BGS.h"
#include "package_bgs/MixtureOfGaussianV2BGS.h"
#include "package_bgs/AdaptiveBackgroundLearning.h"

// The PBAS algorithm was removed from BGSLibrary because it is
// based on patented algorithm ViBE
// http://www2.ulg.ac.be/telecom/research/vibe/
#include "package_bgs/pt/PixelBasedAdaptiveSegmenter.h"

#include <vector>
using namespace std;

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
		
		/// now for MHI and its gangs
		// various tracking parameters (in seconds)
		double MHI_DURATION;
		double MAX_TIME_DELTA;
		double MIN_TIME_DELTA;
		// number of cyclic frame buffer used for motion detection
		// (should, probably, depend on FPS)
		int N;
		
		double pre_ts;
		double pre_reset_ts;
		// ring image buffer
		std::vector<cv::Mat> mhi_buffer;
		int mhi_last;
		
		cv::Mat mhi; // MHI
		cv::Mat mhi_silh, mhi_orient, mhi_mask, mhi_segmask;
		cv::Mat motion;
		cv::Mat trj_history;

		vector < vector<double> > m_left_trajectory;
		vector < vector<double> > m_right_trajectory;
		vector < vector<double> > m_left_lift_to_level;
		vector < vector<double> > m_right_lift_to_level;

		void ResetTrajectory();
		
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
		
		void updateMHI( const cv::Mat& img, cv::Mat& dst, int diff_threshold );
		void updateMHI2( const cv::Mat& img, cv::Mat& dst, int diff_threshold );
  };
}
