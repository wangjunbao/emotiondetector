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
		////std::cout << "max: " << "(" << maxloc.x << "," << maxloc.y << "): " << maxval << std::endl;

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


		return result;

	}


	/* Update locations of subfeatures by running NCC on them*/
	//void updateSubFeatureLocations(string featureFilename, newFaceCoords, boolean doResize, boolean doCrop, int frame #)
	//{
	//	//store face coords
	//	//for now always resize
	//	if(doResize == true or frame # is mod something)
	//	{
	//		resize templates based on new face coords
	//	}
	//	//typically doResize and doCrop are true at same time
	//	for (all sub features)
	//	{
	//		CvRect leftLoc;			
	//		CvRect rightLoc;

	//		CvRect leftSearchSpace;		// with lrBuffer
	//		CvRect rightSearchSpace;
	//		
	//		if(subfeature has no previous coordinates, i.e. == -1)
	//		{
	//			//look at parent feature coords for search space
	//			
	//			//left subtemplate
	//			Mat oldLeftTpl = imread(featureFilename+"Left.jpg",1);
	//			Mat leftTpl; 
	//			resizeFeatureTemplate(oldLeftTpl,newFaceWidth,newFaceHeight,leftTpl);
	//			
	//			leftSearchSpace = cvRect(leftParentLoc.x,leftParentLoc.y,
	//				leftParentLoc.width/2,leftParentLoc.height);

	//			//right subtemplate
	//			Mat oldRightTpl = imread(featureFilename+"Right.jpg",1);
	//			Mat rightTpl; 
	//			resizeFeatureTemplate(oldRightTpl,newFaceWidth,newFaceHeight,rightTpl);
	//			
	//			rightSearchSpace = cvRect(rightParentLoc.x,rightParentLoc.y,
	//				rightParentLoc.width/2,rightParentLoc.height);
	//		}
	//		else
	//		{
	//			//search space = previous loc + lrBuffer + lrMidpt to opposite feature
	//			double lrMidpt = (double)(leftLoc.x + rightLoc.x) / 2.0;
	//			double lrBuffer = lrMidpt / 2.0;
	//			
	//			//dont have checks for boundary conditions
	//			//using only x difference for lrBuffer
	//			leftSearchSpace = cvRect(leftLoc.x - lrBuffer, leftLoc.y - lrBuffer, 
	//				leftLoc.width + 2*lrBuffer, leftLoc.height + 2*lrBuffer);

	//			rightSearchSpace = cvRect(rightLoc.x - lrBuffer, rightLoc.y - lrBuffer, 
	//				rightLoc.width + 2*lrBuffer, rightLoc.height + 2*lrBuffer);
	//		}
	//		//runNCC(search space) and store new maxloc
	//		
	//		//CvRect leftLoc;
	//		bool leftFound = getSearchSpace(img,processedImg,r,leftTpl,leftSearchSpace,leftLoc);
	//		
	//		//CvRect rightLoc;
	//		bool rightFound = getSearchSpace(img,processedImg,r,rightTpl,rightSearchSpace,rightLoc);
	//	}
	//    
	//	if(doCrop == true or frame # is mod something)
	//	{
	//		crop out new template images based on stored locations
	//	}

	//}//end updateSubFeatureLocations


	/* Check if a face found by Haar is valid (has a left eye) */
	boolean isValidFace(IplImage *img, IplImage *processedImg, CvRect *r)
	{
		bool result = false;
		
		double newFaceWidth = r->width;
		double newFaceHeight = r->height;
		
		//look for left eye in face
		Mat oldLeftEyeTpl = imread("templates/leftEye.jpg",1);
		Mat leftEyeTpl;
		resizeFeatureTemplate(oldLeftEyeTpl,newFaceWidth,newFaceHeight,leftEyeTpl);
		CvRect leftEyeSearchSpace = cvRect((r->x), (r->y + r->height/4), r->width/2, (int)((3.0/8.0)*r->height));
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

			//left eye left
			Mat oldLeftEyeLeftTpl = imread("templates/leftEyeLeft.jpg",1);
			Mat leftEyeLeftTpl;
			resizeFeatureTemplate(oldLeftEyeLeftTpl,newFaceWidth,newFaceHeight,leftEyeLeftTpl);
			CvRect leftEyeLeftSearchSpace = cvRect(leftEyeSubSearchSpace.x,leftEyeSubSearchSpace.y,
				leftEyeSubSearchSpace.width/2,leftEyeSubSearchSpace.height);
			CvRect leftEyeLeftSubSearchSpace;
			bool leftEyeLeftFound = getSearchSpace(img,processedImg,r,leftEyeLeftTpl,leftEyeLeftSearchSpace,leftEyeLeftSubSearchSpace);

			//left eye right
			Mat oldLeftEyeRightTpl = imread("templates/leftEyeRight.jpg",1);
			Mat leftEyeRightTpl;
			resizeFeatureTemplate(oldLeftEyeRightTpl,newFaceWidth,newFaceHeight,leftEyeRightTpl);	
			CvRect leftEyeRightSearchSpace = cvRect((leftEyeSubSearchSpace.x + leftEyeSubSearchSpace.width/2),leftEyeSubSearchSpace.y,
				leftEyeSubSearchSpace.width/2,leftEyeSubSearchSpace.height);
			CvRect leftEyeRightSubSearchSpace;
			bool leftEyeRightFound = getSearchSpace(img,processedImg,r,leftEyeRightTpl,leftEyeRightSearchSpace,leftEyeRightSubSearchSpace);

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
			CvRect rightEyeSearchSpace = cvRect((r->x + r->width/2), (r->y + r->height/4), r->width/2, (int)((3.0/8.0)*r->height));
			CvRect rightEyeSubSearchSpace;
			bool rightEyeFound = getSearchSpace(img,processedImg,r,rightEyeTpl,rightEyeSearchSpace,rightEyeSubSearchSpace);

			//left eyebrow
			Mat oldLeftEyebrowTpl = imread("templates/leftEyebrow.jpg",1);
			Mat leftEyebrowTpl;
			resizeFeatureTemplate(oldLeftEyebrowTpl,newFaceWidth,newFaceHeight,leftEyebrowTpl);
			CvRect leftEyebrowSearchSpace = cvRect(r->x, r->y, r->width/2, (int)((3.0/8.0)*r->height));
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
	
};
