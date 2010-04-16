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

	void resizeFeatureTemplate(string filename, double oldFeatureWidth, double oldFeatureHeight, double newFaceWidth, double newFaceHeight, Mat &resizedFeatureImg ) {
		double oldFaceWidth = 228;
		double oldFaceHeight = 228;
		double newFeatureWidth = newFaceWidth * (oldFeatureWidth/oldFaceWidth);
		double newFeatureHeight = newFaceHeight * (oldFeatureHeight/oldFaceHeight);
		
		Mat *featureImg = new Mat;
		*featureImg = imread( "templates/"+filename, 1 );
		
		resize(*featureImg, resizedFeatureImg, Size((int)newFeatureWidth,(int)newFeatureHeight));

		//imwrite("resizedTemplates/"+filename,resizedFeatureImg);
		
		delete featureImg;
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



	/* Check if a face found by Haar is valid (has a left eye) */
	boolean isValidFace(IplImage *img, IplImage *processedImg, CvRect *r)
	{
		bool result = false;
		
		double newFaceWidth = r->width;
		double newFaceHeight = r->height;
		
		//look for left eye in face
		Mat leftEyeTpl;
		resizeFeatureTemplate("leftEye.jpg",61,34,newFaceWidth,newFaceHeight,leftEyeTpl);
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
			Mat leftEyeLeftTpl;
			resizeFeatureTemplate("leftEyeLeft.jpg",10,13,newFaceWidth,newFaceHeight,leftEyeLeftTpl);
			//CvRect leftEyeLeftSearchSpace = cvRect((r->x), (r->y + r->height/4), r->width/2, (int)((3.0/8.0)*r->height));
			CvRect leftEyeLeftSearchSpace = cvRect(leftEyeSubSearchSpace.x,leftEyeSubSearchSpace.y,
				leftEyeSubSearchSpace.width/2,leftEyeSubSearchSpace.height);
			CvRect leftEyeLeftSubSearchSpace;
			bool leftEyeLeftFound = getSearchSpace(img,processedImg,r,leftEyeLeftTpl,leftEyeLeftSearchSpace,leftEyeLeftSubSearchSpace);

			//left eye right
			Mat leftEyeRightTpl;
			resizeFeatureTemplate("leftEyeRight.jpg",7,12,newFaceWidth,newFaceHeight,leftEyeRightTpl);	
			CvRect leftEyeRightSearchSpace = cvRect((leftEyeSubSearchSpace.x + leftEyeSubSearchSpace.width/2),leftEyeSubSearchSpace.y,
				leftEyeSubSearchSpace.width/2,leftEyeSubSearchSpace.height);
			CvRect leftEyeRightSubSearchSpace;
			bool leftEyeRightFound = getSearchSpace(img,processedImg,r,leftEyeRightTpl,leftEyeRightSearchSpace,leftEyeRightSubSearchSpace);


			//right eye
			Mat rightEyeTpl;
			resizeFeatureTemplate("rightEye.jpg",65,37,newFaceWidth,newFaceHeight,rightEyeTpl);
			CvRect rightEyeSearchSpace = cvRect((r->x + r->width/2), (r->y + r->height/4), r->width/2, (int)((3.0/8.0)*r->height));
			CvRect rightEyeSubSearchSpace;
			bool rightEyeFound = getSearchSpace(img,processedImg,r,rightEyeTpl,rightEyeSearchSpace,rightEyeSubSearchSpace);

			//left eyebrow
			Mat leftEyebrowTpl;
			resizeFeatureTemplate("leftEyebrow.jpg",73,29,newFaceWidth,newFaceHeight,leftEyebrowTpl);
			CvRect leftEyebrowSearchSpace = cvRect(r->x, r->y, r->width/2, (int)((3.0/8.0)*r->height));
			CvRect leftEyebrowSubSearchSpace;
			bool leftEyebrowFound = getSearchSpace(img,processedImg,r,leftEyebrowTpl,leftEyebrowSearchSpace,leftEyebrowSubSearchSpace);

			//right eyebrow
			Mat rightEyebrowTpl;
			resizeFeatureTemplate("rightEyebrow.jpg",74,35,newFaceWidth,newFaceHeight,rightEyebrowTpl);
			CvRect rightEyebrowSearchSpace = cvRect((r->x + r->width/2),r->y, r->width/2, (int)((3.0/8.0)*r->height));
			CvRect rightEyebrowSubSearchSpace;
			bool rightEyebrowFound = getSearchSpace(img,processedImg,r,rightEyebrowTpl,rightEyebrowSearchSpace,rightEyebrowSubSearchSpace);

			//mouth
			Mat mouthTpl;
			resizeFeatureTemplate("mouth.jpg",95,53,newFaceWidth,newFaceHeight,mouthTpl);
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
