#pragma once
#include <cv.h>
using namespace cv;

class Face
{

public:
	Face()
	{

	}
	
	Face(CvRect r) 
	{
		this->r = r;
		this->topLeftPoint.x = r.x;
		this->topLeftPoint.y = r.y;
		this->leftEyeTopTpl = imread("templates/leftEyeTop.jpg",1);
		this->leftEyeBottomTpl = imread("templates/leftEyeBottom.jpg",1);
	}
	
	/*
	Face( Point topLeftPoint, Point bottomRightPoint ) {
		this->topLeftPoint = topLeftPoint;
		this->bottomRightPoint = bottomRightPoint;
	}
	*/

	
	Point getTopLeftPoint()
	{
		//return Point(r.x,r.y);
		return topLeftPoint;
	}
	

	/*
	void setTopLeftPoint(Point topLeftPoint)
	{
		this->topLeftPoint = topLeftPoint;
	}

	Point getBottomRightPoint()
	{
		return bottomRightPoint;
	}

	void setBottomRightPoint(Point bottomRightPoint)
	{
		this->bottomRightPoint = bottomRightPoint;
	}
	*/




	Mat getFace()
	{
		return face;
	}

	void setFace(Mat face)
	{
		this->face = face;
	}
	

	void cropTemplates(Mat img, Point location)
	{

	}

	Mat getLeftEye()
	{
		return leftEye;
	}

	void setLeftEye(Mat leftEye)
	{
		this->leftEye = leftEye;
	}

	void resizeFeatureTemplate(Mat& oldFeatureTemplateImg, double newFaceWidth, double newFaceHeight, Mat &resizedFeatureImg )
	{	
		double oldFeatureWidth = (double)oldFeatureTemplateImg.cols;
		double oldFeatureHeight = (double)oldFeatureTemplateImg.rows;
		
		double oldFaceWidth = 228;
		double oldFaceHeight = 228;
			
		double newFeatureWidth = newFaceWidth * (oldFeatureWidth/oldFaceWidth);
		double newFeatureHeight = newFaceHeight * (oldFeatureHeight/oldFaceHeight);
	
		resize(oldFeatureTemplateImg, resizedFeatureImg, Size((int)newFeatureWidth,(int)newFeatureHeight));
	}


	/* Draw a box around an existing face */
	void drawBox(IplImage *img, IplImage *processedImg, CvRect *r)
	{
		//yellow box
		cvRectangle( processedImg,
			cvPoint( r->x, r->y ),
			cvPoint( r->x + r->width, r->y + r->height ),
			CV_RGB( 255, 255, 0 ), 1, 8, 0 );
	}




	bool getSearchSpace(IplImage *img, IplImage *processedImg, CvRect *r, Mat& tpl, CvRect& inputSearchSpace, CvRect& outputSearchSpace)
	{
		if(tpl.empty() == true)
			std::cout << "********************************************TPL IS EMPTY" << std::endl;
		else if (tpl.empty() ==false)
			std::cout << "TPL IS NOT EMPTY" << std::endl;
		bool result = false;
		double THRESH = 0.50;

		//http://nashruddin.com/OpenCV_Region_of_Interest_(ROI)
		
		
		//blue box
		rectangle(Mat(processedImg),Point(inputSearchSpace.x,inputSearchSpace.y),Point(inputSearchSpace.x+inputSearchSpace.width, inputSearchSpace.y+inputSearchSpace.height),CV_RGB(0, 0, 255), 1, 0, 0 );

		cvSetImageROI(img, inputSearchSpace);
		cvSetImageROI(processedImg, inputSearchSpace);

		/* perform template matching */
		Mat res;
		matchTemplate(Mat(img), tpl, res, CV_TM_CCOEFF_NORMED);
		 
		/* find best matches location */
		Point minloc, maxloc;
		double minval = 0.0;
		double maxval = 0.0;

		minMaxLoc(res, &minval, &maxval, &minloc, &maxloc); 

		//green box
		rectangle(Mat(processedImg),maxloc,Point(maxloc.x + tpl.cols, maxloc.y + tpl.rows),CV_RGB(0, 255, 0), 1, 0, 0 );
		//std::cout << "max: " << "(" << maxloc.x << "," << maxloc.y << "): " << maxval << std::endl;

		outputSearchSpace = cvRect(inputSearchSpace.x + maxloc.x, inputSearchSpace.y + maxloc.y, tpl.cols, tpl.rows);

		cvResetImageROI(img);
		cvResetImageROI(processedImg);



		result = maxval > THRESH;

		if(result == true)
		{
			//red box
			cvRectangle( processedImg,
				cvPoint( r->x, r->y ),
				cvPoint( r->x + r->width, r->y + r->height ),
				CV_RGB( 255, 0, 0 ), 1, 8, 0 );
		}


		std::cout << "max: " << "(" << inputSearchSpace.x + maxloc.x << "," << inputSearchSpace.y + maxloc.y << "): " << maxval << std::endl;

		return result;

	}


	/* Update locations of subfeatures by running NCC on them*/
	//void updateSubFeatureLocations(string featureFilename, newFaceCoords, boolean doResize, boolean doCrop, int frame #)
	void updateSubFeatureLocations(IplImage *img, IplImage *processedImg, double newFaceWidth, double newFaceHeight, CvRect& parentLoc)
	{

		//store face coords
		
		//typically doResize and doCrop are true at same time
		CvRect topLoc;			
		CvRect bottomLoc;

		CvRect topSearchSpace;		// if condition is true, parent feature subsection, else subfeature location with Buffer
		CvRect bottomSearchSpace;
		
		//if(subfeature has no previous coordinates, i.e. == -1)
		//if(leftEyeTopTpl.empty() || leftEyeBottomTpl.empty())
		if(true)
		//if(leftEyeTopTpl.empty())
		{
			//look at parent feature coords for search space
			
			//top subtemplate
			//Mat oldTopTpl = imread(featureFilename+"Top.jpg",1);
			Mat oldTopTpl = imread("templates/leftEyeTop.jpg",1);
			//Mat topTpl; 
			//resizeFeatureTemplate(oldTopTpl,newFaceWidth,newFaceHeight,topTpl);
			resizeFeatureTemplate(oldTopTpl,newFaceWidth,newFaceHeight,this->leftEyeTopTpl);
			
			topSearchSpace = cvRect(parentLoc.x, parentLoc.y,parentLoc.width, parentLoc.height/2);

			//bottom subtemplate
			//Mat oldBottomTpl = imread(featureFilename+"Bottom.jpg",1);
			Mat oldBottomTpl = imread("templates/leftEyeBottom.jpg",1);
			//Mat bottomTpl; 
			//resizeFeatureTemplate(oldBottomTpl,newFaceWidth,newFaceHeight,bottomTpl);
			resizeFeatureTemplate(oldBottomTpl,newFaceWidth,newFaceHeight,this->leftEyeBottomTpl);
			
			bottomSearchSpace = cvRect(parentLoc.x, parentLoc.y + parentLoc.height/2, parentLoc.width, parentLoc.height/2);
		}
		else
		{
			////resize sub feature templates and search space based on new face width and height
			//double oldFaceWidth = r.width;
			//double oldFaceHeight = r.height;

			////top
			//double oldTopTplWidth = leftEyeTopTpl.cols;
			//double newTopTplWidth = (oldTopTplWidth / oldFaceWidth) * newFaceWidth;

			//double oldTopTplHeight = leftEyeTopTpl.rows;
			//double newTopTplHeight = (oldTopTplHeight / oldFaceHeight) * newFaceHeight;

			//resize(leftEyeTopTpl,leftEyeTopTpl,Size((int)newTopTplWidth,(int)newTopTplHeight));


			////bottom
			//double oldBottomTplWidth = leftEyeBottomTpl.cols;
			//double newBottomTplWidth = (oldBottomTplWidth / oldFaceWidth) * newFaceWidth;

			//double oldBottomTplHeight = leftEyeBottomTpl.rows;
			//double newBottomTplHeight = (oldBottomTplHeight / oldFaceHeight) * newFaceHeight;

			//resize(leftEyeBottomTpl,leftEyeBottomTpl,Size((int)newBottomTplWidth,(int)newBottomTplHeight));


			////update search space for NCC
			//double newTopY = this->leftEyeTopLoc.y;
			//double newBottomY = this->leftEyeBottomLoc.y;

			//double newMidYDist = (newTopY + newBottomY) / 2.0;

			//int bufferRadius = (int)(newMidYDist / 2.0);

			////we don't have boundary condition checks yet
			//topSearchSpace = cvRect(leftEyeTopLoc.x - bufferRadius, leftEyeTopLoc.y - bufferRadius, 
			//	leftEyeTopTpl.cols + bufferRadius, leftEyeTopTpl.rows + bufferRadius);

			//bottomSearchSpace = cvRect(leftEyeBottomLoc.x - bufferRadius, leftEyeBottomLoc.y - bufferRadius, 
			//	leftEyeBottomTpl.cols + bufferRadius, leftEyeBottomTpl.rows + bufferRadius);

		}//end else


		//update template image by running NCC
		//runNCC(search space) and store new maxloc
		//bool getSearchSpace(IplImage *img, IplImage *processedImg, CvRect *r, Mat& tpl, CvRect& inputSearchSpace, CvRect& outputSearchSpace)
		
		bool topFound = getSearchSpace(img,processedImg,&r,leftEyeTopTpl,topSearchSpace,topLoc);
		
		//update coordinates
		this->leftEyeTopLoc.x = topLoc.x;
		this->leftEyeTopLoc.y = topLoc.y;

		//this->leftEyeTopLoc.x = 50;
		//this->leftEyeTopLoc.y = 50;


		//update template image

		
		////CvRect topLoc;
		//bool topFound = getSearchSpace(img,processedImg,r,topTpl,topSearchSpace,topLoc);
		//
		////CvRect bottomLoc;
		//bool bottomFound = getSearchSpace(img,processedImg,r,bottomTpl,bottomSearchSpace,bottomLoc);

		//if(doCrop == true or frame # is mod something)
		//{
		//	crop out new template images based on stored locations
		//}

	}//end updateSubFeatureLocations




	/* Check if a face found by Haar is valid (has a left eye) */
	boolean isValidFace(IplImage *img, IplImage *processedImg, CvRect *r)
	{
		bool result = false;
		
		double newFaceWidth = r->width;
		double newFaceHeight = r->height;
		
		//look for left eye in face
		Mat oldTopEyeTpl = imread("templates/leftEye.jpg",1);
		Mat leftEyeTpl;
		resizeFeatureTemplate(oldTopEyeTpl,newFaceWidth,newFaceHeight,leftEyeTpl);
		//CvRect leftEyeSearchSpace = cvRect((r->x), (r->y + r->height/4), r->width/2, (int)((3.0/8.0)*r->height));
		//CvRect leftEyeSearchSpace = cvRect((r->x), (r->y + (int)((2.5/8.0)*r->height)), r->width/2, (int)((2.5/8.0)*r->height));
		CvRect leftEyeSearchSpace = cvRect((r->x), (r->y + (int)((2.5/8.0)*r->height)), r->width/2, (int)((2.0/8.0)*r->height));
		CvRect leftEyeSubSearchSpace;
		bool leftEyeFound = getSearchSpace(img,processedImg,r,leftEyeTpl,leftEyeSearchSpace,leftEyeSubSearchSpace);
		

		if(!leftEyeFound)
		{
			result = false;
		}
		else
		{
			//resize each parent feature (except left eye)      
			//each parent feature has specific search space (area of face)
			//run NCC on parent feature (except left eye) to get search space for sub features
			
			//void updateSubFeatureLocations(IplImage *img, IplImage *processedImg, double newFaceWidth, double newFaceHeight, CvRect& parentLoc)
			//updateSubFeatureLocations(img, processedImg, r->width, r->height, leftEyeSubSearchSpace);

			////left eye left
			//Mat oldLeftEyeLeftTpl = imread("templates/leftEyeLeft.jpg",1);
			//Mat leftEyeLeftTpl;
			//resizeFeatureTemplate(oldLeftEyeLeftTpl,newFaceWidth,newFaceHeight,leftEyeLeftTpl);
			//CvRect leftEyeLeftSearchSpace = cvRect(leftEyeSubSearchSpace.x,leftEyeSubSearchSpace.y,
			//	leftEyeSubSearchSpace.width/2,leftEyeSubSearchSpace.height);
			//CvRect leftEyeLeftSubSearchSpace;
			//bool leftEyeLeftFound = getSearchSpace(img,processedImg,r,leftEyeLeftTpl,leftEyeLeftSearchSpace,leftEyeLeftSubSearchSpace);

			////left eye right
			//Mat oldLeftEyeRightTpl = imread("templates/leftEyeRight.jpg",1);
			//Mat leftEyeRightTpl;
			//resizeFeatureTemplate(oldLeftEyeRightTpl,newFaceWidth,newFaceHeight,leftEyeRightTpl);	
			//CvRect leftEyeRightSearchSpace = cvRect((leftEyeSubSearchSpace.x + leftEyeSubSearchSpace.width/2),leftEyeSubSearchSpace.y,
			//	leftEyeSubSearchSpace.width/2,leftEyeSubSearchSpace.height);
			//CvRect leftEyeRightSubSearchSpace;
			//bool leftEyeRightFound = getSearchSpace(img,processedImg,r,leftEyeRightTpl,leftEyeRightSearchSpace,leftEyeRightSubSearchSpace);

			//uncomment starting here:

			//left eye top
			Mat oldLeftEyeTopTpl = imread("templates/leftEyeTop.jpg",1);
			Mat leftEyeTopTpl;
			resizeFeatureTemplate(oldLeftEyeTopTpl,newFaceWidth,newFaceHeight,leftEyeTopTpl);
			CvRect leftEyeTopSearchSpace = cvRect((leftEyeSubSearchSpace.x),(leftEyeSubSearchSpace.y),
				leftEyeSubSearchSpace.width,leftEyeSubSearchSpace.height/2);
			CvRect leftEyeTopSubSearchSpace;
			bool leftEyeTopFound = getSearchSpace(img,processedImg,r,leftEyeTopTpl,leftEyeTopSearchSpace,leftEyeTopSubSearchSpace);

			//left eye bottom
			Mat oldLeftEyeBottomTpl = imread("templates/leftEyeBottom.jpg",1);
			Mat leftEyeBottomTpl;
			resizeFeatureTemplate(oldLeftEyeBottomTpl,newFaceWidth,newFaceHeight,leftEyeBottomTpl);
			CvRect leftEyeBottomSearchSpace = cvRect((leftEyeSubSearchSpace.x),(leftEyeSubSearchSpace.y + leftEyeSubSearchSpace.height/2),
				leftEyeSubSearchSpace.width,leftEyeSubSearchSpace.height/2);
			CvRect leftEyeBottomSubSearchSpace;
			bool leftEyeBottomFound = getSearchSpace(img,processedImg,r,leftEyeBottomTpl,leftEyeBottomSearchSpace,leftEyeBottomSubSearchSpace);


			//right eye
			Mat oldRightEyeTpl = imread("templates/rightEye.jpg",1);
			Mat rightEyeTpl;
			resizeFeatureTemplate(oldRightEyeTpl,newFaceWidth,newFaceHeight,rightEyeTpl);
			//CvRect rightEyeSearchSpace = cvRect((r->x + r->width/2), (r->y + r->height/4), r->width/2, (int)((3.0/8.0)*r->height));
			//CvRect rightEyeSearchSpace = cvRect((r->x + r->width/2), (r->y + (int)((2.5/8.0)*r->height)), r->width/2, (int)((2.5/8.0)*r->height));
			CvRect rightEyeSearchSpace = cvRect((r->x + r->width/2), (r->y + (int)((2.5/8.0)*r->height)), r->width/2, (int)((2.0/8.0)*r->height));
			CvRect rightEyeSubSearchSpace;
			bool rightEyeFound = getSearchSpace(img,processedImg,r,rightEyeTpl,rightEyeSearchSpace,rightEyeSubSearchSpace);

			//left eyebrow
			Mat oldLeftEyebrowTpl = imread("templates/leftEyebrow.jpg",1);
			Mat leftEyebrowTpl;
			resizeFeatureTemplate(oldLeftEyebrowTpl,newFaceWidth,newFaceHeight,leftEyebrowTpl);
			CvRect leftEyebrowSearchSpace = cvRect(r->x, r->y, r->width/2, (int)((3.0/8.0)*r->height));
			//CvRect leftEyebrowSearchSpace = cvRect(r->x, r->y, r->width/2, (int)((4.0/8.0)*r->height));
			CvRect leftEyebrowSubSearchSpace;
			bool leftEyebrowFound = getSearchSpace(img,processedImg,r,leftEyebrowTpl,leftEyebrowSearchSpace,leftEyebrowSubSearchSpace);

			//right eyebrow
			Mat oldRightEyebrowTpl = imread("templates/rightEyebrow.jpg",1);
			Mat rightEyebrowTpl;
			resizeFeatureTemplate(oldRightEyebrowTpl,newFaceWidth,newFaceHeight,rightEyebrowTpl);
			CvRect rightEyebrowSearchSpace = cvRect((r->x + r->width/2),r->y, r->width/2, (int)((3.0/8.0)*r->height));
			CvRect rightEyebrowSubSearchSpace;
			bool rightEyebrowFound = getSearchSpace(img,processedImg,r,rightEyebrowTpl,rightEyebrowSearchSpace,rightEyebrowSubSearchSpace);

			//mouth
			Mat oldMouthTpl = imread("templates/mouth.jpg",1);
			Mat mouthTpl;
			resizeFeatureTemplate(oldMouthTpl,newFaceWidth,newFaceHeight,mouthTpl);
			CvRect mouthSearchSpace = cvRect(r->x, (r->y +  (int)((3.0/8.0)*r->height)), r->width, (int)((5.0/8.0)*r->height));
			CvRect mouthSubSearchSpace;
			bool mouthFound = getSearchSpace(img,processedImg,r,mouthTpl,mouthSearchSpace,mouthSubSearchSpace);


			result = true;
		}

		return result;
	}




private:
	//templates:
	CvRect r;
	Mat face;
	
	Mat leftEye;

	Point topLeftPoint;
	//Point bottomRightPoint;

	Point leftEyeTopLoc;
	Point leftEyeBottomLoc;

	Mat leftEyeTopTpl;
	Mat leftEyeBottomTpl;
	
};
