
#include "FrameProcessor.h"
#include <fstream>
using namespace std;
using namespace cv;

namespace ibb
{
  FrameProcessor::FrameProcessor() : firstTime(true), frameNumber(0), duration(0), tictoc(""), frameToStop(0)
  {
    std::cout << "FrameProcessor()" << std::endl;

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
		sprintf(tmp, "%s/bin%06d.jpg", savePath.c_str(), frameNumber);
#endif
		saveName = tmp;		
	}

	if(probPath.length() > 0)
	{
		char prob[512];
#if defined(_WIN32)		
		sprintf(prob, "%s\\prob%06d.jpg", probPath.c_str(), frameNumber);
#else		
		sprintf(prob, "%s/prob%06d.jpg", probPath.c_str(), frameNumber);
#endif		
		probName = prob;
	}

	if(csvPath.length() > 0)
	{
		char csv[512];
#if defined(_WIN32)		
		sprintf(csv, "%s\\%d.csv", csvPath.c_str(), frameNumber);
#else		
		sprintf(csv, "%s/%d.csv", csvPath.c_str(), frameNumber);
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
				rectangle(img_prep, face_i, CV_RGB(0, 255,0), 1);
//				string gender;
			}
		}
		
		cv::imshow("Pre Processor", img_prep);
		
    firstTime = false;
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
