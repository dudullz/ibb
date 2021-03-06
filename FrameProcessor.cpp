
#include "FrameProcessor.h"
#include <fstream>
#include <string>
using namespace std;
using namespace cv;

namespace ibb
{
  FrameProcessor::FrameProcessor() : firstTime(true), frameNumber(0), duration(0), tictoc(""), frameToStop(0)
  {
    std::cout << "FrameProcessor()" << std::endl;

	m_show_debug_info = false;

	MHI_DURATION = 2;
	MAX_TIME_DELTA = 0.3;
	MIN_TIME_DELTA = 0.05;
	N = 30;

	m_sw_duration = 2;	// check 2 seconds window
	m_valid_perc = 0.6;	// for 90% of time
	m_silh_threshold = 0.001;
	m_motion_threshold = 0.001;
	
	m_empty_scene = false;

	m_score_lh_d2m = 0.0;		// left hand down to middle
	m_score_lh_m2u = 0.0;		// left hand middle to up
	m_score_lh_d2u = 0.0;		// left hand down to up
	m_score_lh_u2m = 0.0;		// left hand up to middle
	m_score_lh_m2d = 0.0;		// left hand middle to down
	m_score_lh_u2d = 0.0;		// left hand up to down

	m_score_rh_d2m = 0.0;		// right hand down to middle
	m_score_rh_m2u = 0.0;		// right hand middle to up
	m_score_rh_d2u = 0.0;		// right hand down to up
	m_score_rh_u2m = 0.0;		// right hand up to middle
	m_score_rh_m2d = 0.0;		// right hand middle to down
	m_score_rh_u2d = 0.0;		// right hand up to down	 

	m_count_leftup = m_count_leftdown = m_count_rightup = m_count_rightdown = 0;
	m_count_valid_threshold = 5;

    loadConfig();
    saveConfig();
#ifdef __APPLE__
		LoadModelLeftHandUpCLib();
		LoadModelLeftHandDownCLib();
		LoadModelRightHandUpCLib();
		LoadModelRightHandDownCLib();
#else
	LoadModelLeftHandUp();
	LoadModelLeftHandDown();
	LoadModelRightHandUp();
	LoadModelRightHandDown();
#endif
  }

  FrameProcessor::~FrameProcessor()
  {
    std::cout << "~FrameProcessor()" << std::endl;
  }

  void FrameProcessor::init()
  {
	   std::cout << " [FrameProcessor::init()] " << std::endl;	
    if (enablePreProcessor)
	{
		std::cout << "********* Use PreProcess"  << std::endl;
      preProcessor = new PreProcessor;
	}

    if (enableWeightedMovingMeanBGS)
      weightedMovingMean = new WeightedMovingMeanBGS;

    if (enableWeightedMovingVarianceBGS)
      weightedMovingVariance = new WeightedMovingVarianceBGS;

    if (enableMixtureOfGaussianV1BGS)
      mixtureOfGaussianV1BGS = new MixtureOfGaussianV1BGS;

    if (enableMixtureOfGaussianV2BGS)
      mixtureOfGaussianV2BGS = new MixtureOfGaussianV2BGS;

    if (enableAdaptiveBackgroundLearning)
      adaptiveBackgroundLearning = new AdaptiveBackgroundLearning;

    if(enablePBAS)
	{
		std::cout << "********* Use PBAS"  << std::endl;
      pixelBasedAdaptiveSegmenter = new PixelBasedAdaptiveSegmenter;
	}
		
		/// we are going to perform MHI anyways, so the relevant variables are initilised here
		/// even the face detection and UpperBodyDetector is actually setup after this init()
		mhi_buffer.resize(N);
		mhi_last = 0;

		m_lefthand_trajectory.clear();
		m_righthand_trajectory.clear();
  }
	
	void FrameProcessor::setUpperBodyDetector(std::string ub_model)
	{		
		fpUBModel = ub_model;
		overlapThreshold = 0.2;
		numThreads = 2;
		// current version of LatentSvmDetector takes vector of model files as inputs,
		// so that it's possible to detect more than one class 
		std::vector<std::string> models;
		models.push_back(fpUBModel);
		lsvm_detector.load( models );
		enableUpperBodayDetector = true;
	
		generateColors( colors, lsvm_detector.getClassNames().size() );
	}
	
	void FrameProcessor::setFaceDetector(std::string face_model)
	{
		fpFaceModel = face_model;
		face_detector.load( fpFaceModel );
		enableFaceDetector = true;
	}
	
  void FrameProcessor::process(std::string name, IBGS *bgs, const cv::Mat &img_input, cv::Mat &img_bgs)
  {
    if (tictoc == name)
      tic(name);

    cv::Mat img_bkgmodel;
    bgs->process(img_input, img_bgs, img_bkgmodel);

    if (tictoc == name)
      toc();
  }

  void FrameProcessor::process(const cv::Mat &img_input)
  {
	  frameNumber++;
	  std::cout << "	[[[ FrameProcessor::process(...) at Frame " << frameNumber << " ]]]" << std::endl;	    
	
	if(savePath.length() > 0)
	{
		char tmp[512];
#if defined(_WIN32)
		sprintf(tmp, "%s\\bin%06d.jpg", savePath.c_str(), frameNumber);		
#else
		sprintf(tmp, "%s/bin%06ld.jpg", savePath.c_str(), frameNumber);
#endif
		saveName = tmp;		
	}

	if(probPath.length() > 0)
	{
		char prob[512];
#if defined(_WIN32)		
		sprintf(prob, "%s\\prob%06d.jpg", probPath.c_str(), frameNumber);
#else		
		sprintf(prob, "%s/prob%06ld.jpg", probPath.c_str(), frameNumber);
#endif		
		probName = prob;
	}

	if(csvPath.length() > 0)
	{
		char csv[512];
#if defined(_WIN32)		
		sprintf(csv, "%s\\%d.csv", csvPath.c_str(), frameNumber);
#else		
		sprintf(csv, "%s/%ld.csv", csvPath.c_str(), frameNumber);
#endif		
		csvName = csv;
	}

    if (enablePreProcessor)
      preProcessor->process(img_input, img_prep);

    if (enableWeightedMovingMeanBGS)
      process("WeightedMovingMeanBGS", weightedMovingMean, img_prep, img_wmovmean);

    if (enableWeightedMovingVarianceBGS)
      process("WeightedMovingVarianceBGS", weightedMovingVariance, img_prep, img_movvar);

    if (enableMixtureOfGaussianV1BGS)
	{
      process("MixtureOfGaussianV1BGS", mixtureOfGaussianV1BGS, img_prep, img_mog1);	  
		if(saveName.length() > 0 )
		{
			imwrite(saveName, img_mog1);
			std::cout << "Save: " << saveName << std::endl;
		}
	}

    if (enableMixtureOfGaussianV2BGS)
	{
      process("MixtureOfGaussianV2BGS", mixtureOfGaussianV2BGS, img_prep, img_mog2);
	  if(saveName.length() > 0 )
		{
			imwrite(saveName, img_mog2);
			std::cout << "Save: " << saveName << std::endl;
		}
	}

    if (enableAdaptiveBackgroundLearning)
      process("AdaptiveBackgroundLearning", adaptiveBackgroundLearning, img_prep, img_bkgl_fgmask);

    if(enablePBAS)
	{
	  // img_pt_pbas is foreground mask
      process("PBAS", pixelBasedAdaptiveSegmenter, img_prep, img_pt_pbas);
	  if(saveName.length() > 0 )
		{
			imwrite(saveName, img_pt_pbas);
			std::cout << "Save: " << saveName << std::endl;
		}
	  if(probName.length() > 0)
	  {
			pixelBasedAdaptiveSegmenter->getProbMap(img_pt_pbas_prob);
			imwrite(probName, img_pt_pbas_prob);
			std::cout << "Save: " << probName << std::endl;

			if(csvName.length() > 0)
			{
				writeProbToCSV(img_pt_pbas_prob);
				std::cout << "Save: " << csvName << std::endl;
			}
	  }
	}

   		if( enableUpperBodayDetector )
		{
			cout << "1.   Run Upper Body Detector (overlapThreshold: " << overlapThreshold << ", numThreads: " << numThreads << ")" << endl;
			cout << fpUBModel << endl;
			vector<cv::LatentSvmDetector::ObjectDetection> detections;
			lsvm_detector.detect( img_input, detections, overlapThreshold, numThreads);
			cout << "Detections: " << detections.size() << endl;
			
//			const vector<string>& classNames = lsvm_detector.getClassNames();
//			cout << "Loaded " << classNames.size() << " models:" << endl;
//			for( size_t i = 0; i < classNames.size(); i++ )
//			{
//        cout << i << ") " << classNames[i] << "; ";
//			}
			
			for( size_t i = 0; i < detections.size(); i++ )
			{
        const LatentSvmDetector::ObjectDetection& od = detections[i];
				rectangle( img_prep, od.rect, colors[od.classID], 3 );
			}
			// put text over the all rectangles
//			for( size_t i = 0; i < detections.size(); i++ )
//			{
//        const LatentSvmDetector::ObjectDetection& od = detections[i];
//        putText( img_input, classNames[od.classID], Point(od.rect.x+4,od.rect.y+13), FONT_HERSHEY_SIMPLEX, 0.55, colors[od.classID], 2 );
//			}
		}
		
		if( enableFaceDetector )
		{
			cout << "2.   Run Face Detection" << endl;
			cout << fpFaceModel << endl;
			vector< Rect_<int> > faces;
			Mat grey;
			cvtColor(img_input, grey, CV_BGR2GRAY);
			face_detector.detectMultiScale(grey, faces);
			cout << "Faces: " << faces.size() << endl;
			for(int i = 0; i < faces.size(); i++)
			{
				// Process face by face:
				Rect face_i = faces[i];
				// Crop the face from the image. So simple with OpenCV C++:
//				Mat face = gray(face_i);
				// Resizing the face is necessary for Eigenfaces and Fisherfaces. You can easily
				// verify this, by reading through the face recognition tutorial coming with OpenCV.
				// Resizing IS NOT NEEDED for Local Binary Patterns Histograms, so preparing the
				// input data really depends on the algorithm used.
				//
				// I strongly encourage you to play around with the algorithms. See which work best
				// in your scenario, LBPH should always be a contender for robust face recognition.
				//
				// Since I am showing the Fisherfaces algorithm here, I also show how to resize the
				// face you have just found:
//				Mat face_resized;
//				cv::resize(face, face_resized, Size(im_width, im_height), 1.0, 1.0, INTER_CUBIC);
				// Now perform the prediction, see how easy that is:
//				int prediction = model->predict(face_resized);
				// And finally write all we've found out to the original image!
				// First of all draw a green rectangle around the detected face:
				rectangle(img_prep, face_i, CV_RGB(255,0,0), 4);
//				string gender;
			}
		}
		
		/// By now we should have the correction positon of the stadning person (through the deteced face
		/// and/or the upper body detection result)
		cout << "3.   Detect Human Action" << endl;
		RecogniseAction(img_input, motion);
		imshow( "Motion", motion );
		imshow( "Motion History", trj_history );					

//********************** For Demo Purpose **********************//
//////////////////////////////////////////////////////////////////
//		char rlts[128];
//		sprintf(rlts, "性别：男（98%）\n年龄：30~35岁 (95%)");
//		putText( img_prep, rlts, Point(10,20), FONT_HERSHEY_SIMPLEX, 0.55, CV_RGB(0,255,0), 2 );
		
//		cv::imshow("Pre Processor", img_prep);
		
		//char rlts[128];
		//sprintf(rlts, "Gender: Male");
		//putText( img_prep, rlts, Point(10,20), FONT_HERSHEY_SIMPLEX, 0.85, CV_RGB(0,255,0), 2 );
		//sprintf(rlts, "Age: 30-35");
		//putText( img_prep, rlts, Point(10,50), FONT_HERSHEY_SIMPLEX, 0.85, CV_RGB(0,255,0), 2 );

		//sprintf(rlts, "FPS:10.29");
		//putText( img_prep, rlts, Point(500,30), FONT_HERSHEY_SIMPLEX, 0.85, CV_RGB(0,0,255), 5 );
		//putText( img_prep, rlts, Point(500,30), FONT_HERSHEY_SIMPLEX, 0.85, CV_RGB(255,255,255), 1 );

		//int x = img_prep.cols / 3 - 15;
		//Point convex[4];
		//convex[0].x = x;	convex[0].y = 15; 
		//convex[1].x = img_prep.cols - x + 30;	convex[1].y = 15;
		//convex[2].x = img_prep.cols - x + 30;	convex[2].y = 55;
		//convex[3].x = x;	convex[3].y = 55;
		//fillConvexPoly(img_prep, convex, 4, CV_RGB(255,255,255));
		//sprintf(rlts, "Left Arm Lifted");
		//putText( img_prep, rlts, Point(200,45), FONT_HERSHEY_SIMPLEX, 1.00, CV_RGB(255,0,128), 3 );
//////////////////////////////////////////////////////////////////					
		cv::imshow("Pre Processor", img_prep);	

    firstTime = false;
  }

  void FrameProcessor::CalcSlidingWindowParas()
  {
	  m_sw_length = m_sw_duration * m_fps;
	  m_sw_valid_length = m_sw_length * m_valid_perc;
	  cout << "m_sw_length: " << m_sw_length << ", m_sw_valid_length: " << m_sw_valid_length << endl;
	  char temp[128];
	  sprintf(temp, "FPS:%2.f, m_sw_length:%d, m_sw_valid_length:%ld", m_fps, m_sw_length, m_sw_valid_length);
	  putText(img_prep, temp, Point(10, img_prep.rows - 20), FONT_HERSHEY_SIMPLEX, 0.55, CV_RGB(0, 0, 255), 2);
	  if (m_show_debug_info)
	  {
		  sprintf(temp, "m_motionless_count:%d, m_motion_count:%d", m_motionless_count, m_motion_count);
		  putText(img_prep, temp, Point(10, img_prep.rows - 40), FONT_HERSHEY_SIMPLEX, 0.55, CV_RGB(0, 0, 255), 2);
	  }

	  // at the same time draw a cross help to locate the actor
	  line(img_prep, Point(0, img_prep.rows/2), Point(img_prep.cols, img_prep.rows/2), CV_RGB(255, 0, 0), 2);
	  line(img_prep, Point(img_prep.cols/2, 0), Point(img_prep.cols/2, img_prep.rows), CV_RGB(255, 0, 0), 2);
  }
	
  void FrameProcessor::RecogniseAction(const cv::Mat& img, cv::Mat& dst)
  {
	  UpdateMHI(img, dst, 30);
	  CalcSlidingWindowParas();
	  if (m_show_debug_info)
	  {
		  cout << "m_silh_ratio: " << m_silh_ratio << ", m_silh_threshold: " << m_silh_threshold << endl;
		  cout << "LefhandUp Range: [" << m_modlen_lh_d2m / 2 << ",  " << m_modlen_lh_d2m * 1.5 << " ]" << endl;
		  cout << "LefhandDown Range: [" << m_modlen_lh_m2d / 2 << ",  " << m_modlen_lh_m2d * 1.5 << " ]" << endl;
		  cout << "RighthandUp Range: [" << m_modlen_rh_d2m / 2 << ",  " << m_modlen_rh_d2m * 1.5 << " ]" << endl;
		  cout << "RighthandDown Range: [" << m_modlen_rh_m2d / 2 << ",  " << m_modlen_rh_m2d * 1.5 << " ]" << endl;
	  }

	  if (m_silh_ratio < m_silh_threshold)
	  {
		  m_motion_count = 0;
		  m_motionless_count++;
		  if (m_motionless_count > m_sw_valid_length && !m_empty_scene)
		  {			 
			  m_empty_scene = true;
			  ResetTrajectory();
			  trj_history = Mat::zeros(mhi_mask.size(), CV_8UC3);
		  }
	  }
	  else{
		  /// FIXME  should also check for 'valid' motions from different image regions
		  m_empty_scene = false;
		  m_motionless_count = 0;
		  m_motion_count++;
		  int llen = m_lefthand_trajectory.size();		  
		  if (m_modlen_lh_d2m / 2 < llen && llen < m_modlen_lh_d2m * 1.5)
		  {
			  cout << "\n	*** Check if Lefthand Up ***" << endl;
			  int rdim = m_lefthand_trajectory[0].size();
			  cout << m_model_lefthand_down2middle.size() << " VS " << m_lefthand_trajectory.size() << endl;
			  m_dtw_left.Initialise(m_model_lefthand_down2middle, m_lefthand_trajectory, rdim);
			  m_dtw_left.ComputeLoaclCostMatrix();
			  m_score_lh_d2m = m_dtw_left.DTWDistance1Step();
			  cout << "	>>>  Left Hand Up Probability: " << m_score_lh_d2m << endl;

			  m_dtw_left.Release();
		  }

		  if (llen > m_modlen_lh_m2d / 2 && llen < m_modlen_lh_m2d * 1.5)
		  {
			  cout << "\n	*** Check if Lefthand Down ***" << endl;
			  int rdim = m_lefthand_trajectory[0].size();
			  cout << m_model_lefthand_middle2down.size() << " VS " << m_lefthand_trajectory.size() << endl;
			  m_dtw_left.Initialise(m_model_lefthand_middle2down, m_lefthand_trajectory, rdim);
			  m_dtw_left.ComputeLoaclCostMatrix();
			  m_score_lh_m2d = m_dtw_left.DTWDistance1Step();
			  cout << "	>>>  Left Hand Down Probability: " << m_score_lh_m2d << endl;

			  m_dtw_left.Release();
		  }

		  int rlen = m_righthand_trajectory.size();
		  if (m_modlen_rh_d2m / 2 < rlen && rlen < m_modlen_rh_d2m * 1.5)
		  {
			  cout << "\n	*** Check if Righthand Up ***" << endl;
			  int rdim = m_righthand_trajectory[0].size();
			  cout << m_model_righthand_down2middle.size() << " VS " << m_righthand_trajectory.size() << endl;
			  m_dtw_right.Initialise(m_model_righthand_down2middle, m_righthand_trajectory, rdim);
			  m_dtw_right.ComputeLoaclCostMatrix();
			  m_score_rh_d2m = m_dtw_right.DTWDistance1Step();
			  cout << "	>>>  Right Hand Up Probability: " << m_score_rh_d2m << endl;

			  m_dtw_right.Release();
		  }

		  if (m_modlen_rh_m2d / 2 < rlen && rlen < m_modlen_rh_m2d * 1.5)
		  {
			  cout << "\n	*** Check if Righthand Down ***" << endl;
			  int rdim = m_righthand_trajectory[0].size();
			  cout << m_model_righthand_middle2down.size() << " VS " << m_righthand_trajectory.size() << endl;
			  m_dtw_right.Initialise(m_model_righthand_middle2down, m_righthand_trajectory, rdim);
			  m_dtw_right.ComputeLoaclCostMatrix();
			  m_score_rh_m2d = m_dtw_right.DTWDistance1Step();
			  cout << "	>>>  Right Hand Down Probability: " << m_score_rh_m2d << endl;

			  m_dtw_right.Release();
		  }

	  }  

	  char k;
	  k = waitKey(10);
	  if (k == 'q')
		  exit(1);
	  if (k == 'c')
	  {
		  cout << "	>>>>>>>>>> Trajecotry Cleared <<<<<<<<<<" << endl;
		  ResetTrajectory();
	  }

	  if (k == 'l')
	  {
		  cout << "	@@@@@@@@@@@ Save Lefthand Up Model @@@@@@@@@@@" << endl;
		  std::ofstream csv;
		  csv.open("LeftUp.model", std::ofstream::out | std::ofstream::trunc);

		  int llen = m_lefthand_trajectory.size();
		  int dim = m_lefthand_trajectory[0].size();

		  csv << llen << " " << dim << endl;
		  for (int i = 0; i < llen; ++i)
		  {
			  if (dim == 1)
				  csv << m_lefthand_trajectory[i][0];
			  else
			  {
				  for (int d = 0; d < dim; ++d)
				  {
					  if (d != (dim - 1))
						  csv << m_lefthand_trajectory[i][d] << ",";
					  else
						  csv << m_lefthand_trajectory[i][d];
				  }
			  }
			  csv << "\n";
		  }
		  csv.close();

		  m_model_lefthand_down2middle = m_lefthand_trajectory;
		  //m_lefthand_trajectory.clear();
	  }

	  if (k == 'd')
	  {
		  cout << "	@@@@@@@@@@@ Save Lefthand Down Model @@@@@@@@@@@" << endl;
		  std::ofstream csv;
		  csv.open("LeftDown.model", std::ofstream::out | std::ofstream::trunc);

		  int llen = m_lefthand_trajectory.size();
		  int dim = m_lefthand_trajectory[0].size();

		  csv << llen << " " << dim << endl;
		  for (int i = 0; i < llen; ++i)
		  {
			  if (dim == 1)
				  csv << m_lefthand_trajectory[i][0];
			  else
			  {
				  for (int d = 0; d < dim; ++d)
				  {
					  if (d != (dim - 1))
						  csv << m_lefthand_trajectory[i][d] << ",";
					  else
						  csv << m_lefthand_trajectory[i][d];
				  }
			  }
			  csv << "\n";
		  }
		  csv.close();

		  m_model_lefthand_middle2down = m_lefthand_trajectory;
		  //m_lefthand_trajectory.clear();
	  }

	  if (k == 'r')
	  {
		  cout << "	@@@@@@@@@@@ Save Righthand Up Model @@@@@@@@@@@" << endl;
		  std::ofstream csv;
		  csv.open("RightUp.model", std::ofstream::out | std::ofstream::trunc);

		  int rlen = m_righthand_trajectory.size();
		  int dim = m_righthand_trajectory[0].size();

		  csv << rlen << " " << dim << endl;
		  for (int i = 0; i < rlen; ++i)
		  {
			  if (dim == 1)
				  csv << m_righthand_trajectory[i][0];
			  else
			  {
				  for (int d = 0; d < dim; ++d)
				  {
					  if (d != (dim - 1))
						  csv << m_righthand_trajectory[i][d] << ",";
					  else
						  csv << m_righthand_trajectory[i][d];
				  }
			  }
			  csv << "\n";
		  }
		  csv.close();

		  m_model_righthand_down2middle = m_righthand_trajectory;
		  //m_righthand_trajectory.clear();
	  }

	  if (k == 'w')
	  {
		  cout << "	@@@@@@@@@@@ Save Righthand Down Model @@@@@@@@@@@" << endl;
		  std::ofstream csv;
		  csv.open("RightDown.model", std::ofstream::out | std::ofstream::trunc);

		  int rlen = m_righthand_trajectory.size();
		  int dim = m_righthand_trajectory[0].size();

		  csv << rlen << " " << dim << endl;
		  for (int i = 0; i < rlen; ++i)
		  {
			  if (dim == 1)
				  csv << m_righthand_trajectory[i][0];
			  else
			  {
				  for (int d = 0; d < dim; ++d)
				  {
					  if (d != (dim - 1))
						  csv << m_righthand_trajectory[i][d] << ",";
					  else
						  csv << m_righthand_trajectory[i][d];
				  }
			  }
			  csv << "\n";
		  }
		  csv.close();

		  m_model_righthand_middle2down = m_righthand_trajectory;
		  //m_righthand_trajectory.clear();
	  }

	  if (k == 'v')
	  {
		  cout << "Calculate Probability: " << endl;

		  int rdim = m_righthand_trajectory[0].size();
		  cout << m_model_righthand_down2middle.size() << " VS " << m_righthand_trajectory.size() << endl;
		  m_dtw_right.Initialise(m_model_righthand_down2middle, m_righthand_trajectory, rdim);
		  m_dtw_right.ComputeLoaclCostMatrix();
		  m_score_rh_d2m = m_dtw_right.DTWDistance1Step();
		  cout << "	>>>  Right Hand Up Probability: " << m_score_rh_d2m << endl;

		  m_dtw_right.Release();
	  }

	  if (k == 'a')
	  {
		  cout << "Calculate Probability: " << endl;
		  int ldim = m_lefthand_trajectory[0].size();
		  m_dtw_left.Initialise(m_model_lefthand_down2middle, m_lefthand_trajectory, ldim);
		  m_dtw_left.ComputeLoaclCostMatrix();
		  m_score_lh_d2m = m_dtw_left.DTWDistance1Step();
		  cout << "Left Hand Up Probability: " << m_score_lh_d2m << endl;

		  m_dtw_left.Release();
	  }

	  char str_trj[128];
	  sprintf(str_trj, "Left Traj Num:%ld", m_lefthand_trajectory.size());
	  putText(img_prep, str_trj, Point(10, 20), FONT_HERSHEY_SIMPLEX, 0.55, CV_RGB(0, 255, 0), 2);

	  if (0 < m_score_lh_d2m && m_score_lh_d2m < 500)
	  {
		  m_count_leftup++;
		  sprintf(str_trj, "Left Hand Up!");
		  putText(img_prep, str_trj, Point(10, 60), FONT_HERSHEY_SIMPLEX, 0.95, CV_RGB(255, 0, 0), 2);
	  }
	  else
	  {
		  sprintf(str_trj, "LeftUp Prob: %.2f", m_score_lh_d2m);
		  putText(img_prep, str_trj, Point(10, 60), FONT_HERSHEY_SIMPLEX, 0.55, CV_RGB(0, 255, 0), 2);
	  }

	  if (0 < m_score_lh_m2d && m_score_lh_m2d < 500)
	  {
		  m_count_leftdown++;
		  sprintf(str_trj, "Left Hand Down!");
		  putText(img_prep, str_trj, Point(10, 100), FONT_HERSHEY_SIMPLEX, 1.0, CV_RGB(255, 0, 0), 2);
	  }
	  else
	  {
		  sprintf(str_trj, "LeftDown Prob: %.2f", m_score_lh_m2d);
		  putText(img_prep, str_trj, Point(10, 100), FONT_HERSHEY_SIMPLEX, 0.55, CV_RGB(0, 255, 0), 2);
	  }

	  int xRight = 2 * img_prep.cols / 3;
	  sprintf(str_trj, "Right Traj Num:%ld", m_righthand_trajectory.size());
	  putText(img_prep, str_trj, Point(xRight, 20), FONT_HERSHEY_SIMPLEX, 0.55, CV_RGB(0, 255, 0), 2);
	  	  
	  if (0.0 < m_score_rh_d2m && m_score_rh_d2m < 500)
	  {
		  m_count_rightup++;
		  sprintf(str_trj, "Right Hand Up!");
		  putText(img_prep, str_trj, Point(xRight - 80, 60), FONT_HERSHEY_SIMPLEX, 1.0, CV_RGB(255, 0, 0), 2);
	  }
	  else
	  {
		  sprintf(str_trj, "RightUp Prob: %.2f", m_score_rh_d2m);
		  putText(img_prep, str_trj, Point(xRight, 60), FONT_HERSHEY_SIMPLEX, 0.55, CV_RGB(0, 255, 0), 2);
	  }

	  if (0.0 < m_score_rh_m2d && m_score_rh_m2d < 500)
	  {
		  m_count_rightdown++;
		  sprintf(str_trj, "Right Hand Down!");
		  putText(img_prep, str_trj, Point(xRight - 80, 100), FONT_HERSHEY_SIMPLEX, 1.0, CV_RGB(255, 0, 0), 2);
	  }
	  else
	  {
		  sprintf(str_trj, "RightDown Prob: %.2f", m_score_rh_m2d);
		  putText(img_prep, str_trj, Point(xRight, 100), FONT_HERSHEY_SIMPLEX, 0.55, CV_RGB(0, 255, 0), 2);
	  }

	  cout << "	< Left Trajectory: " << m_lefthand_trajectory.size() << " >" << endl;
	  cout << "LeftUp Model: " << m_model_lefthand_down2middle.size() << endl;
	  cout << "LeftDown Model: " << m_model_lefthand_middle2down.size() << endl;

	  cout << "	< Right Trajectory: " << m_righthand_trajectory.size() << " >" << endl;
	  cout << "RightUp Model: " << m_model_righthand_down2middle.size() << endl;
	  cout << "RightDown Model: " << m_model_righthand_middle2down.size() << endl;

	  if (m_motionless_count >= m_count_valid_threshold)
	  {
		  if (m_count_leftup >= m_count_valid_threshold || m_count_leftdown >= m_count_valid_threshold)
		  {
			  m_count_leftup = m_count_leftdown = 0;
			  m_lefthand_trajectory.clear();
			  m_score_lh_d2m = m_score_lh_m2d = 0.0;
		  }
		  if (m_count_rightup >= m_count_valid_threshold || m_count_rightdown >= m_count_valid_threshold)
		  {
			  m_count_rightup = m_count_rightdown = 0;
			  m_righthand_trajectory.clear();
			  m_score_rh_d2m = m_score_rh_m2d = 0.0;
		  }
	  }
  }
  void FrameProcessor::ResetTrajectory()
  {
	  //vector::clear() does not free memory allocated by the vector to store objects; it calls destructors for the objects it holds.
	  //The vector has to manage storage internally for the objects it stores.Creating a new vector requires allocating new storage, but clearing & reusing an existing vector allows(but doesn't guarantee) reuse of its already-allocated storage. 
	  //If you call clear (or a resize with smaller size) on a vector of anything, then all elements from that vector which need to be deleted have their destructors called and their memory is released.
	  //If you have a vector of vectors, then each inner vector's destructor will clean up its resources properly. When a row-vector or column-vector is destroyed, it cleans up after itself automatically.
	  //Actual "memory management" is supposed to be abstracted away by std::vector. What's important is that after clear the objects are destroyed and the memory is, well... Not released as in "operator delete", but released as in "it's now available to be reused by new objects in the vector or returned to the operating system", which is whas I tried to say here- indeed imprecisely.
	  m_lefthand_trajectory.clear();
	  m_righthand_trajectory.clear();

	  m_score_lh_d2m = 0.0;		// left hand down to middle
	  m_score_lh_m2u = 0.0;		// left hand middle to up
	  m_score_lh_d2u = 0.0;		// left hand down to up
	  m_score_lh_u2m = 0.0;		// left hand up to middle
	  m_score_lh_m2d = 0.0;		// left hand middle to down
	  m_score_lh_u2d = 0.0;		// left hand up to down

	  m_score_rh_d2m = 0.0;		// right hand down to middle
	  m_score_rh_m2u = 0.0;		// right hand middle to up
	  m_score_rh_d2u = 0.0;		// right hand down to up
	  m_score_rh_u2m = 0.0;		// right hand up to middle
	  m_score_rh_m2d = 0.0;		// right hand middle to down
	  m_score_rh_u2d = 0.0;		// right hand up to down	 

	  m_count_leftup = m_count_leftdown = m_count_rightup = m_count_rightdown = 0;
  }

  void FrameProcessor::UpdateMHI( const cv::Mat &img, cv::Mat &dst, int diff_threshold)
	{
		double timestamp = (double)clock()/CLOCKS_PER_SEC; // get current time in seconds
		int idx1 = mhi_last, idx2;
		idx2 = (mhi_last + 1) % N; // index of (last - (N-1))th frame
		mhi_last = idx2;
		cvtColor( img, mhi_buffer[mhi_last], CV_BGR2GRAY ); // convert frame to grayscale
				
		printf("ts: %.2f\n", timestamp);
		printf("idx1: %d, idx2: %d\n", idx1, idx2);
		
		if( mhi_buffer[idx1].size() != mhi_buffer[idx2].size() )
			mhi_silh = Mat::ones(img.size(), CV_8U)*255;
		else
			absdiff(mhi_buffer[idx1], mhi_buffer[idx2], mhi_silh); // get difference between frames

//		imshow( "Silh", mhi_silh );
		threshold( mhi_silh, mhi_silh, diff_threshold, 1, CV_THRESH_BINARY ); // and threshold it
		
		/// make a vertical black stripe to the image
		/// make sure that the motions on each side are not connected!
		int height = img.rows;
		int width = 80;
		int x = img.cols / 2 - width / 2;
		int y = 0;
		
		cv::Rect roi(x, y, width, height);
		cv::Mat vertical_bar = mhi_silh( roi );
		subtract(vertical_bar, vertical_bar, vertical_bar);
//		imshow("mask_bar", vertical_bar);
		
		if( mhi.empty() )
			mhi = Mat::zeros(mhi_silh.size(), CV_32F);
		updateMotionHistory( mhi_silh, mhi, timestamp, MHI_DURATION ); // update MHI
//		imshow("mhi", mhi);
		
		int nc = norm(mhi_silh, NORM_L1); // calculate number of points within silhouette ROI
		int nz = countNonZero(mhi_silh);		
		m_silh_ratio = (double)nz / (double)mhi_silh.total();
		cout << "Norm: " << nc << ", Motion: " << nz << ", Image Area: " << mhi_silh.total() << " - ratio:" << m_silh_ratio << endl;

		// convert MHI to blue 8u image
		mhi.convertTo(mhi_mask, CV_8U, 255./MHI_DURATION,
									(MHI_DURATION - timestamp)*255./MHI_DURATION);
		
		dst = Mat::zeros(mhi_mask.size(), CV_8UC3);
		if( trj_history.empty() )
			trj_history = Mat::zeros(mhi_mask.size(), CV_8UC3);
		if( (timestamp - pre_reset_ts) > 10000.0 )
		{
			trj_history = Mat::zeros(mhi_mask.size(), CV_8UC3);
			//ResetTrajectory();
			pre_reset_ts = timestamp;
		}
		
		insertChannel(mhi_mask, dst, 0);
//		imshow( "mask", mhi_mask );
		// calculate motion gradient orientation and valid orientation mask
		calcMotionGradient( mhi, mhi_mask, mhi_orient, MAX_TIME_DELTA, MIN_TIME_DELTA, 3 );
		//imshow("mhi_orient", mhi_mask);	// the mask here should only contain 1,2,3,... small integers???
		// segment motion: get sequence of motion components
		// segmask is marked motion components map. It is not used further
		vector<Rect> brects;
		segmentMotion(mhi, mhi_segmask, brects, timestamp, MAX_TIME_DELTA );
		
		/// NOTE: there seems to be a bug in mhi_segmask where
		/// after a little while it will be cleared!!!
		imshow( "segmask", mhi_segmask );
		
		// iterate through the motion components,
		// One more iteration (i == -1) corresponds to the whole image (global motion)
		int numTooSmall = 0;
		for( int i = -1; i < (int)brects.size(); i++ ) {
			Rect roi; Scalar color; double magnitude;
			Mat maski = mhi_mask;
			if( i < 0 ) { // case of the whole image
				roi = Rect(0, 0, img.cols, img.rows);
				color = Scalar::all(255);
				magnitude = 100;
			}
			else { // i-th motion component
				roi = brects[i];
				if( roi.area() < 3000 ) // reject very small components
				{
					numTooSmall++;
					continue;
				}
				color = Scalar(0, 0, 255);
				magnitude = 30;
				maski = mhi_mask(roi);
			}

			if (m_show_debug_info)
				printf("	ROI %d: (%d,%d)-%dx%d-%d\n", i, roi.x, roi.y, roi.width, roi.height, roi.area());
//			if( i == 0)
//				imshow("Maski", maski);
			// calculate orientation
			double angle = calcGlobalOrientation( mhi_orient(roi), maski, mhi(roi), timestamp, MHI_DURATION);
			angle = 360.0 - angle;  // adjust for images with top-left origin
						
			int count = norm( mhi_silh, NORM_L1 ); // calculate number of points within silhouette ROI

			if (m_show_debug_info)
				cout << "count: " << count << endl;
			// check for the case of little motion
			if( count < roi.area() * 0.05 )
				continue;
			
			// starts to log movement direction
			// just consider 1 dimension for now
			if (i >= 0)
			{
				vector<double> item;
				item.push_back(angle);
				if (roi.x < img.cols / 2)
					m_lefthand_trajectory.push_back(item);
				else if (roi.x > img.cols / 2)
					m_righthand_trajectory.push_back(item);
			}
			
			char temp[64];
				sprintf(temp, "Angle:%.2f", angle);
			if (m_show_debug_info)
			{					
				if (i >= 0)
					printf("%s, xPos:%d\n", temp, roi.x);
			}
			
			// draw a clock with arrow indicating the direction
			Point center( roi.x + roi.width/2, roi.y + roi.height/2 );
			circle( dst, center, cvRound(magnitude*1.2), color, 3, CV_AA, 0 );
			line( dst, center, Point( cvRound( center.x + magnitude*cos(angle*CV_PI/180)),
															 cvRound( center.y - magnitude*sin(angle*CV_PI/180))), color, 3, CV_AA, 0 );
			
			line( trj_history, center, Point( cvRound( center.x + magnitude*cos(angle*CV_PI/180)),
															 cvRound( center.y - magnitude*sin(angle*CV_PI/180))), color, 3, CV_AA, 0 );
			putText( dst, temp, center, FONT_HERSHEY_SIMPLEX, 0.55, color, 2 );
			
		}
		if (m_show_debug_info)
			printf("Motion Component Number: %ld, Too small: %d\n", brects.size(), numTooSmall);
		
		m_fps = 1 / (timestamp - pre_ts);
		pre_ts = timestamp;
		printf("fps: %.2f\n", m_fps);
	}
		
  void FrameProcessor::writeProbToCSV(cv::Mat &in_prob_map)
  {
	  // accept only char type matrices
	  CV_Assert(in_prob_map.depth() != sizeof(uchar));

	  int channels = in_prob_map.channels();
	  int nRows = in_prob_map.rows;
	  int nCols = in_prob_map.cols;
	  std::cout << "Original channels: " << channels << ", nRows:" << nRows << ", nCols:" << nCols << std::endl;

	  cv::Mat prob_map;
	  if(channels == 1)
		  prob_map = in_prob_map;
	  /// For MultiLayer & FSOM specifically, both save greyscale image using RGB format (equal R,G,B values)
	  else if(channels == 3)
		  cv::cvtColor(in_prob_map, prob_map, CV_BGR2GRAY);  
	  
	  /// It is essential that the new line is added at end of each row in the resulting CSV file
	  //if (prob_map.isContinuous())
	  //{
		 // nCols *= nRows;
		 // nRows = 1;

		 // std::cout << "Save in Memory Continuously: " << nCols << "x" << nRows << std::endl;
	  //}

		std::ofstream csv;
	  csv.open(csvName.c_str());

	  int i,j;
	  uchar* p;
	  csv.setf( std::ios::fixed, std::ios::floatfield ); // floatfield set to fixed
	  csv.precision(2);

		for( i = 0; i < nRows; ++i)
		{
			p = prob_map.ptr<uchar>(i);
			for ( j = 0; j < nCols; ++j)
			{
				//printf("%d:%.2f\t", p[j], p[j]);
				float prob = (float)p[j] / 255.0f;
				if( j != nCols - 1 )
					csv << prob << ",";
				else
					csv << prob;
			}
			csv << "\n";
		}
  
	  csv.close();
  }

  void FrameProcessor::finish(void)
  {
    /*if(enableMultiLayerBGS)
    multiLayerBGS->finish();

    if(enableLBSimpleGaussian)
    lbSimpleGaussian->finish();

    if(enableLBFuzzyGaussian)
    lbFuzzyGaussian->finish();

    if(enableLBMixtureOfGaussians)
    lbMixtureOfGaussians->finish();

    if(enableLBAdaptiveSOM)
    lbAdaptiveSOM->finish();

    if(enableLBFuzzyAdaptiveSOM)
    lbFuzzyAdaptiveSOM->finish();*/

    if(enablePBAS)
      delete pixelBasedAdaptiveSegmenter;

    if (enableAdaptiveBackgroundLearning)
      delete adaptiveBackgroundLearning;

    if (enableMixtureOfGaussianV2BGS)
      delete mixtureOfGaussianV2BGS;

    if (enableMixtureOfGaussianV1BGS)
      delete mixtureOfGaussianV1BGS;

    if (enableWeightedMovingVarianceBGS)
      delete weightedMovingVariance;

    if (enableWeightedMovingMeanBGS)
      delete weightedMovingMean;

    if (enablePreProcessor)
      delete preProcessor;
  }

  void FrameProcessor::tic(std::string value)
  {
    processname = value;
    duration = static_cast<double>(cv::getTickCount());
  }

  void FrameProcessor::toc()
  {
    duration = (static_cast<double>(cv::getTickCount()) - duration) / cv::getTickFrequency();
    std::cout << processname << "\ttime(sec):" << std::fixed << std::setprecision(6) << duration << std::endl;
  }

  void FrameProcessor::saveConfig()
  {	  
#if defined(_WIN32)
	//CvFileStorage* fs = cvOpenFileStorage("F:\\Developer\\BGS\\AndrewsSobral\\bgslibrary\\config\\FrameProcessor.xml", 0, CV_STORAGE_WRITE);
	CvFileStorage* fs = cvOpenFileStorage("config\\FrameProcessor.xml", 0, CV_STORAGE_WRITE);
#else
    CvFileStorage* fs = cvOpenFileStorage("./config/FrameProcessor.xml", 0, CV_STORAGE_WRITE);
#endif
    
    cvWriteString(fs, "tictoc", tictoc.c_str());

    cvWriteInt(fs, "enablePreProcessor", enablePreProcessor);
    cvWriteInt(fs, "enableWeightedMovingMeanBGS", enableWeightedMovingMeanBGS);
    cvWriteInt(fs, "enableWeightedMovingVarianceBGS", enableWeightedMovingVarianceBGS);
    cvWriteInt(fs, "enableMixtureOfGaussianV1BGS", enableMixtureOfGaussianV1BGS);
    cvWriteInt(fs, "enableMixtureOfGaussianV2BGS", enableMixtureOfGaussianV2BGS);
    cvWriteInt(fs, "enableAdaptiveBackgroundLearning", enableAdaptiveBackgroundLearning);

    cvWriteInt(fs, "enablePBAS", enablePBAS);
    cvReleaseFileStorage(&fs);
  }

  void FrameProcessor::loadConfig()
  {
#if defined(_WIN32)
	//CvFileStorage* fs = cvOpenFileStorage("F:\\Developer\\BGS\\AndrewsSobral\\bgslibrary\\config\\FrameProcessor.xml", 0, CV_STORAGE_READ);
	CvFileStorage* fs = cvOpenFileStorage("config\\FrameProcessor.xml", 0, CV_STORAGE_READ);
#else
    CvFileStorage* fs = cvOpenFileStorage("./config/FrameProcessor.xml", 0, CV_STORAGE_READ);
#endif

    tictoc = cvReadStringByName(fs, 0, "tictoc", "");

    enablePreProcessor = cvReadIntByName(fs, 0, "enablePreProcessor", true);
    enableWeightedMovingMeanBGS = cvReadIntByName(fs, 0, "enableWeightedMovingMeanBGS", false);
    enableWeightedMovingVarianceBGS = cvReadIntByName(fs, 0, "enableWeightedMovingVarianceBGS", false);
    enableMixtureOfGaussianV1BGS = cvReadIntByName(fs, 0, "enableMixtureOfGaussianV1BGS", false);
    enableMixtureOfGaussianV2BGS = cvReadIntByName(fs, 0, "enableMixtureOfGaussianV2BGS", false);
    enableAdaptiveBackgroundLearning = cvReadIntByName(fs, 0, "enableAdaptiveBackgroundLearning", false);
    enablePBAS = cvReadIntByName(fs, 0, "enablePBAS", false);

    cvReleaseFileStorage(&fs);
  }

  void FrameProcessor::LoadModelLeftHandUp()
  {
	  cout << "Load Lefthand Up Model " << endl;
	  std::ifstream file;

	  //file.open("LeftUp.model", std::ifstream::in);
		
//		cout << "3" << endl;
//		if (file.good())
//		{
//			cout << "The file not good" << endl;
//			exit(-1);
//		}
//		
//		cout << "1" << endl;
//		if (file.fail())
//		{
//			cout << "The file doesn't exist" << endl;
//			exit(-1);
//		}		
//		
//		cout << "2" << endl;
//		if (file.is_open())
//		{
//			cout << "Error opening the file" << endl;
//			exit(-1);
//		}
		
//	  string line;
//	  while (getline(file, line))
//			cout << line << endl;

	 // double a = -1.0;
	 // while (!file.eof())
	 // {
		//  file >> a;
		//  cout << a << endl;
	 // }
		//getchar();
	 // file.close();

	  file.open("LeftUp.model", std::ifstream::in);
	  //file.seekg(0);
	  int count = 0;
	  int dims = -1;
	  double value;
	  while (!file.eof())
	  {
		  if (count == 0)
		  {
			  file >> m_modlen_lh_d2m;
			  
			  cout << "Length: " << m_modlen_lh_d2m << endl;
		  }
		  else if (count == 1)
		  {
			  file >> dims;
			  if (dims > 1)
			  {
				  for (int i = 0; i < m_modlen_lh_d2m; ++i)
				  {
					  m_model_lefthand_down2middle[i].assign(dims, 0.0);
				  }
			  }
			  cout << "Dim: " << dims << endl;
		  }
		  else{
			  file >> value;
			  cout << "value: " << value << endl;
			  vector<double> item;
			  item.push_back(value);
			  m_model_lefthand_down2middle.push_back(item);
		  }

		  count++;
	  }
	  m_model_lefthand_down2middle.erase(m_model_lefthand_down2middle.end()-1);
	  cout << "Verify length: " << m_model_lefthand_down2middle.size() << endl;	  
	  for (int i = 0; i < m_model_lefthand_down2middle.size(); ++i)
		  cout << i + 1 << ":	" << m_model_lefthand_down2middle[i][0] << endl;

	  getchar();

	  file.close();
  }

  void FrameProcessor::LoadModelLeftHandDown()
  {
	  cout << "Load Lefthand Downd Model " << endl;
	  std::ifstream file;

	  //file.open("LeftDown.model", std::ifstream::in);
	  //double a = -1.0;
	  //while (!file.eof())
	  //{
		 // file >> a;
		 // cout << a << endl;
	  //}
	  //getchar();
	  //file.close();

	  file.open("LeftDown.model", std::ifstream::in);
	  //file.seekg(0, file.beg);
	  int count = 0;
	  int dims = -1;
	  double value;
	  while (!file.eof())
	  {
		  if (count == 0)
		  {
			  file >> m_modlen_lh_m2d;

			  cout << "Length: " << m_modlen_lh_m2d << endl;
		  }
		  else if (count == 1)
		  {
			  file >> dims;
			  if (dims > 1)
			  {
				  for (int i = 0; i < m_modlen_lh_m2d; ++i)
				  {
					  m_model_lefthand_middle2down[i].assign(dims, 0.0);
				  }
			  }
			  cout << "Dim: " << dims << endl;
		  }
		  else{
			  file >> value;
			  cout << "value: " << value << endl;
			  vector<double> item;
			  item.push_back(value);
			  m_model_lefthand_middle2down.push_back(item);
		  }

		  count++;
	  }
	  m_model_lefthand_middle2down.erase(m_model_lefthand_middle2down.end() - 1);
	  cout << "Verify length: " << m_model_lefthand_middle2down.size() << endl;
	  for (int i = 0; i < m_model_lefthand_middle2down.size(); ++i)
		  cout << i + 1 << ":	" << m_model_lefthand_middle2down[i][0] << endl;

	  getchar();

	  file.close();
  }

  void FrameProcessor::LoadModelRightHandUp()
  {
	  cout << "Load Righthand Up Model" << endl;
	  std::ifstream file;

	 // file.open("RightUp.model", std::ifstream::in);
	 // string line;
	 // //while (getline(file, line))
	 // // cout << line << endl;

	 // double a = -1;
	 // while (!file.eof())
	 // {
		//  file >> a;
		//  cout << a << endl;
	 // }
		//getchar();
	 // file.close();

	  file.open("RightUp.model", std::ifstream::in);
	  //file.seekg(0, file.beg);
	  int count = 0;
	  int dims = -1;
	  double value;
	  while (!file.eof())
	  {
		  if (count == 0)
		  {
			  file >> m_modlen_rh_d2m;

			  cout << "Length: " << m_modlen_rh_d2m << endl;
		  }
		  else if (count == 1)
		  {
			  file >> dims;
			  if (dims > 1)
			  {
				  for (int i = 0; i < m_modlen_rh_d2m; ++i)
				  {
					  m_model_righthand_down2middle[i].assign(dims, 0.0);
				  }
			  }
			  cout << "Dim: " << dims << endl;
		  }
		  else{
			  file >> value;
			  cout << "value: " << value << endl;
			  vector<double> item;
			  item.push_back(value);
			  m_model_righthand_down2middle.push_back(item);
		  }

		  count++;
	  }
	  m_model_righthand_down2middle.erase(m_model_righthand_down2middle.end() - 1);
	  cout << "Verify length: " << m_model_righthand_down2middle.size() << endl;
	  for (int i = 0; i < m_model_righthand_down2middle.size(); ++i)
		  cout << i + 1 << ":	" << m_model_righthand_down2middle[i][0] << endl;

	  getchar();

	  file.close();
  }

  void FrameProcessor::LoadModelRightHandDown()
  {
	  cout << "Load Righthand Down Model" << endl;
	  std::ifstream file;

	  //file.open("RightDown.model", std::ifstream::in);
	  //string line;
	  ////while (getline(file, line))
	  //// cout << line << endl;

	  //double a = -1;
	  //while (!file.eof())
	  //{
		 // file >> a;
		 // cout << a << endl;
	  //}
	  //getchar();
	  //file.close();

	  file.open("RightDown.model", std::ifstream::in);
	  //file.seekg(0, file.beg);
	  int count = 0;
	  int dims = -1;
	  double value;
	  while (!file.eof())
	  {
		  if (count == 0)
		  {
			  file >> m_modlen_rh_m2d;

			  cout << "Length: " << m_modlen_rh_m2d << endl;
		  }
		  else if (count == 1)
		  {
			  file >> dims;
			  if (dims > 1)
			  {
				  for (int i = 0; i < m_modlen_rh_m2d; ++i)
				  {
					  m_model_righthand_middle2down[i].assign(dims, 0.0);
				  }
			  }
			  cout << "Dim: " << dims << endl;
		  }
		  else{
			  file >> value;
			  cout << "value: " << value << endl;
			  vector<double> item;
			  item.push_back(value);
			  m_model_righthand_middle2down.push_back(item);
		  }

		  count++;
	  }
	  m_model_righthand_middle2down.erase(m_model_righthand_middle2down.end() - 1);
	  cout << "Verify length: " << m_model_righthand_middle2down.size() << endl;
	  for (int i = 0; i < m_model_righthand_middle2down.size(); ++i)
		  cout << i + 1 << ":	" << m_model_righthand_middle2down[i][0] << endl;

	  getchar();

	  file.close();
  }
	
	void FrameProcessor::LoadModelLeftHandUpCLib()
	{
		FILE *ifp;
		ifp = fopen("LeftUp.model", "r");
		
		if (ifp == NULL) {
			fprintf(stderr, "Can't open input file LeftUp.model!\n");
			exit(1);
		}
		
		int dims = -1;
		fscanf(ifp, "%d %d", &m_modlen_lh_d2m, &dims);
		cout << "Read in Length:" << m_modlen_lh_d2m << ", Dim:" << dims << endl;
		
		double value = -1.0;
		while ( !feof(ifp) )
		{
			if (fscanf(ifp, "%lf", &value) != 1)
				break;
			cout << value << endl;
			vector<double> item;
			item.push_back(value);
			m_model_lefthand_down2middle.push_back(item);
		}
		cout << "Verify length: " << m_model_lefthand_down2middle.size() << endl;
		
		fclose(ifp);
	}

	void FrameProcessor::LoadModelLeftHandDownCLib()
	{
		FILE *ifp;
		ifp = fopen("LeftDown.model", "r");

		if (ifp == NULL) {
			fprintf(stderr, "Can't open input file LeftDown.model!\n");
			exit(1);
		}

		int dims = -1;
		fscanf(ifp, "%d %d", &m_modlen_lh_m2d, &dims);
		cout << "Read in Length:" << m_modlen_lh_m2d << ", Dim:" << dims << endl;

		double value = -1.0;
		while (!feof(ifp))
		{
			if (fscanf(ifp, "%lf", &value) != 1)
				break;
			cout << value << endl;
			vector<double> item;
			item.push_back(value);
			m_model_lefthand_middle2down.push_back(item);
		}
		cout << "Verify length: " << m_model_lefthand_middle2down.size() << endl;

		fclose(ifp);
	}
	
	void FrameProcessor::LoadModelRightHandUpCLib()
	{
		FILE *ifp;
		ifp = fopen("RightUp.model", "r");
		
		if (ifp == NULL) {
			fprintf(stderr, "Can't open input file RightUp.model!\n");
			exit(1);
		}
		
		int dims = -1;
		fscanf(ifp, "%d %d", &m_modlen_rh_d2m, &dims);
		cout << "Read in Length:" << m_modlen_rh_d2m << ", Dim:" << dims << endl;
		
		double value = -1.0;
		while ( !feof(ifp) )
		{
			if (fscanf(ifp, "%lf", &value) != 1)
				break;
			cout << value << endl;
			vector<double> item;
			item.push_back(value);
			m_model_righthand_down2middle.push_back(item);
		}
		cout << "Verify length: " << m_model_righthand_down2middle.size() << endl;
		
		fclose(ifp);
	}

	void FrameProcessor::LoadModelRightHandDownCLib()
	{
		FILE *ifp;
		ifp = fopen("RightDown.model", "r");

		if (ifp == NULL) {
			fprintf(stderr, "Can't open input file RightDown.model!\n");
			exit(1);
		}

		int dims = -1;
		fscanf(ifp, "%d %d", &m_modlen_rh_m2d, &dims);
		cout << "Read in Length:" << m_modlen_rh_m2d << ", Dim:" << dims << endl;

		double value = -1.0;
		while (!feof(ifp))
		{
			if (fscanf(ifp, "%lf", &value) != 1)
				break;
			cout << value << endl;
			vector<double> item;
			item.push_back(value);
			m_model_righthand_middle2down.push_back(item);
		}
		cout << "Verify length: " << m_model_righthand_middle2down.size() << endl;

		fclose(ifp);
	}
}


 