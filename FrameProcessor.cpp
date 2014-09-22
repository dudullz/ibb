
#include "FrameProcessor.h"
#include <fstream>
using namespace std;
using namespace cv;

namespace ibb
{
  FrameProcessor::FrameProcessor() : firstTime(true), frameNumber(0), duration(0), tictoc(""), frameToStop(0)
  {
    std::cout << "FrameProcessor()" << std::endl;

	MHI_DURATION = 2;
	MAX_TIME_DELTA = 0.3;
	MIN_TIME_DELTA = 0.05;
	N = 30;

    loadConfig();
    saveConfig();
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

		m_left_trajectory.clear();
		m_right_trajectory.clear();
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
	  std::cout << " [FrameProcessor::process(const cv::Mat &img_input)] " << std::endl;	
    frameNumber++;
	
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
		cout << "3.   Run MHI Detection" << endl;
		updateMHI( img_input, motion, 30 );
		imshow( "Motion", motion );
//		imshow( "Motion History", trj_history );
		
		char str_trj[128];
		sprintf(str_trj, "Left Traj Num:%ld", m_left_trajectory.size());
		putText(img_prep, str_trj, Point(10, 20), FONT_HERSHEY_SIMPLEX, 0.55, CV_RGB(0, 255, 0), 2);
		sprintf(str_trj, "Right Traj Num:%ld", m_right_trajectory.size());
		int xRight = 2 * img_prep.cols / 3;
		putText(img_prep, str_trj, Point(xRight, 20), FONT_HERSHEY_SIMPLEX, 0.55, CV_RGB(0, 255, 0), 2);
			

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
		imwrite("LeftUp.jpg", img_prep);

    firstTime = false;
  }
	
  void FrameProcessor::ResetTrajectory()
  {
	  //vector::clear() does not free memory allocated by the vector to store objects; it calls destructors for the objects it holds.
	  //The vector has to manage storage internally for the objects it stores.Creating a new vector requires allocating new storage, but clearing & reusing an existing vector allows(but doesn't guarantee) reuse of its already-allocated storage. 
	  //If you call clear (or a resize with smaller size) on a vector of anything, then all elements from that vector which need to be deleted have their destructors called and their memory is released.
	  //If you have a vector of vectors, then each inner vector's destructor will clean up its resources properly. When a row-vector or column-vector is destroyed, it cleans up after itself automatically.
	  //Actual "memory management" is supposed to be abstracted away by std::vector. What's important is that after clear the objects are destroyed and the memory is, well... Not released as in "operator delete", but released as in "it's now available to be reused by new objects in the vector or returned to the operating system", which is whas I tried to say here- indeed imprecisely.
	  m_left_trajectory.clear();
	  m_right_trajectory.clear();
	 
  }

	void FrameProcessor::updateMHI( const cv::Mat &img, cv::Mat &dst, int diff_threshold)
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
		
		// convert MHI to blue 8u image
		mhi.convertTo(mhi_mask, CV_8U, 255./MHI_DURATION,
									(MHI_DURATION - timestamp)*255./MHI_DURATION);
		
		dst = Mat::zeros(mhi_mask.size(), CV_8UC3);
		if( trj_history.empty() )
			trj_history = Mat::zeros(mhi_mask.size(), CV_8UC3);
		if( (timestamp - pre_reset_ts) > 100.0 )
		{
			trj_history = Mat::zeros(mhi_mask.size(), CV_8UC3);
			ResetTrajectory();
			pre_reset_ts = timestamp;
		}
		
		insertChannel(mhi_mask, dst, 0);
//		imshow( "mask", mhi_mask );
		// calculate motion gradient orientation and valid orientation mask
		calcMotionGradient( mhi, mhi_mask, mhi_orient, MAX_TIME_DELTA, MIN_TIME_DELTA, 3 );
//		imshow("mhi_orient", mhi_mask);
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
			printf("	ROI %d: (%d,%d)-%dx%d-%d\n", i, roi.x, roi.y, roi.width, roi.height, roi.area());
//			if( i == 0)
//				imshow("Maski", maski);
			// calculate orientation
			double angle = calcGlobalOrientation( mhi_orient(roi), maski, mhi(roi), timestamp, MHI_DURATION);
			angle = 360.0 - angle;  // adjust for images with top-left origin
						
			int count = norm( mhi_silh, NORM_L1 ); // calculate number of points within silhouette ROI
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
					m_left_trajectory.push_back(item);
				else if (roi.x > img.cols / 2)
					m_right_trajectory.push_back(item);
			}
			
			char temp[64];
			sprintf(temp, "Angle:%.2f", angle);
			if (i >= 0)
				printf("%s, xPos:%d\n", temp, roi.x);
			
			// draw a clock with arrow indicating the direction
			Point center( roi.x + roi.width/2, roi.y + roi.height/2 );
			circle( dst, center, cvRound(magnitude*1.2), color, 3, CV_AA, 0 );
			line( dst, center, Point( cvRound( center.x + magnitude*cos(angle*CV_PI/180)),
															 cvRound( center.y - magnitude*sin(angle*CV_PI/180))), color, 3, CV_AA, 0 );
			
			line( trj_history, center, Point( cvRound( center.x + magnitude*cos(angle*CV_PI/180)),
															 cvRound( center.y - magnitude*sin(angle*CV_PI/180))), color, 3, CV_AA, 0 );
			putText( dst, temp, center, FONT_HERSHEY_SIMPLEX, 0.55, color, 2 );
			
		}
		printf("Motion Component Number: %ld, Too small: %d\n", brects.size(), numTooSmall);
		
		double fps = 1 / (timestamp - pre_ts);
		pre_ts = timestamp;
		printf("fps: %.2f\n", fps);
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
}
