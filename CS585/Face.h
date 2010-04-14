#pragma once
#include <cv.h>
using namespace cv;

class Face
{

public:
	Face( Point topLeftPoint, Point bottomRightPoint ) {
		this->topLeftPoint = topLeftPoint;
		this->bottomRightPoint = bottomRightPoint;
	}

	Point getTopLeftPoint()
	{
		return topLeftPoint;
	}

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

//when creating a face
	boolean isValidFace(IplImage *img, IplImage *processedImg, double newFaceWidth, double newFaceHeight)
	{
		double THRESH = 0.50;
		Mat tpl;
		resizeFeatureTemplate("lefteye.jpg",61,34,newFaceWidth,newFaceHeight,tpl);
		
		//std::cout << "tpl: " << tpl.cols << "," << tpl.rows << std::endl;

		//http://nashruddin.com/OpenCV_Region_of_Interest_(ROI)
		
		//CvRect rect = cvRect((r->x), (r->y + r->height/4), r->width/2, (int)((3.0/8.0)*r->height));
	


		CvRect rect = cvRect((topLeftPoint.x), (topLeftPoint.y + (bottomRightPoint.y - topLeftPoint.y)/4), (bottomRightPoint.x-topLeftPoint.x)/2, (int)((3.0/8.0)*(bottomRightPoint.y - topLeftPoint.y)));
		rectangle(Mat(processedImg),Point(rect.x,rect.y),Point(rect.x+rect.width, rect.y+rect.height),CV_RGB(0, 0, 255), 1, 0, 0 );

		cvSetImageROI(img, rect);
		cvSetImageROI(processedImg, rect);
		
		Mat res;
		 
		/* perform template matching */
		matchTemplate(Mat(img), tpl, res, CV_TM_CCOEFF_NORMED);
		 
		/* find best matches location */
		Point minloc, maxloc;
		double minval = 0.0;
		double maxval = 0.0;

		minMaxLoc(res, &minval, &maxval, &minloc, &maxloc); 

		if (maxval <= THRESH)
		{
			return false;
		}
		else
		{
			//resize each parent feature (except left eye)
	  //      
			////each parent feature has specific search space (area of face)
			//run NCC on parent feature (except left eye) to get search space for sub features
	        
			rectangle(Mat(processedImg),maxloc,Point(maxloc.x + tpl.cols, maxloc.y + tpl.rows),CV_RGB(0, 255, 0), 1, 0, 0 );
			std::cout << "max: " << "(" << maxloc.x << "," << maxloc.y << "): " << maxval << std::endl;


						//cvResetImageROI(img);
			//cvResetImageROI(processedImg);
			


			return true;
		}

	}
private:
	//templates:
	
	Mat face;
	
	Mat leftEye;

	Point topLeftPoint;
	Point bottomRightPoint;
	
};
