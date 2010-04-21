//#pragma once
#ifndef FACE
#define FACE
#include <cv.h>
//#include <math.h>
using namespace cv;

class Face
{
private:
	//templates:
	CvRect r;

	int oldFaceWidth;
	int oldFaceHeight;

	int currentFaceWidth;
	int currentFaceHeight;


	Mat face;
	
	Mat leftEye;

	Point topLeftPoint;
	//Point bottomRightPoint;

	Point leftEyeTopLoc;
	Point leftEyeBottomLoc;

	Mat leftEyeTopTpl;
	Mat leftEyeBottomTpl;

public:
	//Face()
	//{

	//}
	
	Face(CvRect r) 
	{
		this->r = r;
		this->topLeftPoint.x = r.x;
		this->topLeftPoint.y = r.y;

		this->currentFaceWidth = r.width;
		this->currentFaceHeight = r.height;

		//this->leftEyeTopTpl = Mat(0,0,CV_8UC3);
		//this->leftEyeBottomTpl = Mat(0,0,CV_8UC3);
		
		/*
		std::cout << "left rows before: " << leftEyeTopTpl.rows << std::endl;
		//this->leftEyeTopTpl = imread("templates/leftEyeTop.jpg",1);
		std::cout << "left rows after: " << leftEyeTopTpl.rows << std::endl;
		this->leftEyeBottomTpl = imread("templates/leftEyeBottom.jpg",1);
		*/
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

	int getWidth()
	{
		return currentFaceWidth;
	}

	int getHeight()
	{
		return currentFaceHeight;
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
	//bool getSearchSpace(IplImage *img, IplImage *processedImg, CvRect& r, Mat& tpl, CvRect& inputSearchSpace, CvRect& outputSearchSpace)
	{
		/*
		if(tpl.empty() == true)
			std::cout << "********************************************TPL IS EMPTY" << std::endl;
		else if (tpl.empty() ==false)
			std::cout << "TPL IS NOT EMPTY" << std::endl;
		*/
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
		//std::cout << "**************outputSearchSpace: " << outputSearchSpace.width << " by " << outputSearchSpace.height << std::endl;

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

			//cvRectangle( processedImg,
			//	cvPoint( r.x, r.y ),
			//	cvPoint( r.x + r.width, r.y + r.height ),
			//	CV_RGB( 255, 0, 0 ), 1, 8, 0 );
		}


		//std::cout << "max: " << "(" << inputSearchSpace.x + maxloc.x << "," << inputSearchSpace.y + maxloc.y << "): " << maxval << std::endl;

		return result;

	}

	void updateMouthSubFeatureLocs(IplImage *img, IplImage *processedImg, CvRect& r, CvRect& parentLoc)
	{
		//update face with new coordinates
		this->oldFaceWidth = this->currentFaceWidth;
		this->oldFaceHeight = this->currentFaceWidth;

		this->currentFaceWidth = r.width;	//newFaceWidth
		this->currentFaceHeight = r.height;	//newFaceHeight
		
		this->topLeftPoint.x = r.x;
		this->topLeftPoint.y = r.y;


		CvRect topLoc = cvRect(0,0,0,0);			
		CvRect bottomLoc = cvRect(0,0,0,0);

		CvRect topSearchSpace;		// if condition is true, parent feature subsection, else subfeature location with Buffer
		CvRect bottomSearchSpace;

		//subfeature has no previous coordinates, i.e. the template images are empty
		if(leftEyeTopTpl.empty() || leftEyeBottomTpl.empty())
		//if(true)
		{
			//look at parent feature coords for search space
			
			//crop out eye template
			Rect parentROI(parentLoc);	//Make a rectangle
			Mat imgParentROI = Mat(img)(parentROI);	//Point a cv::Mat header at it (no allocation is done)
			//imgParentROI.copyTo(this->leftEyeParentTpl);

			cvNamedWindow( "eyeTop", 1 );
			imshow( "eyeTop", imgParentROI);


			//top subtemplate
			//Mat oldTopTpl = imread("templates/leftEyeTop.jpg",1);			
			//resizeFeatureTemplate(oldTopTpl,r.width,r.height,this->leftEyeTopTpl);
			
			//topSearchSpace = cvRect(parentLoc.x, parentLoc.y,parentLoc.width, parentLoc.height/2);
			topSearchSpace = cvRect(parentLoc.x, parentLoc.y,parentLoc.width, (3.0/8.0)*parentLoc.height);


			Rect topROI(topSearchSpace);
			Mat imgTopROI = Mat(img)(topROI);
			imgTopROI.copyTo(this->leftEyeTopTpl);

			cvNamedWindow( "leftEyeTop", 1 );
			imshow( "leftEyeTop", this->leftEyeTopTpl);

			
			//yellow box
			//rectangle(Mat(processedImg),Point(topSearchSpace.x,topSearchSpace.y),
			//	Point(topSearchSpace.x+topSearchSpace.width,topSearchSpace.y+topSearchSpace.height),
			//	CV_RGB(255,255,0),1,0,0);


			//bottom subtemplate
			//Mat oldBottomTpl = imread("templates/leftEyeBottom.jpg",1);
			//resizeFeatureTemplate(oldBottomTpl,r.width,r.height,this->leftEyeBottomTpl);

			//bottomSearchSpace = cvRect(parentLoc.x, parentLoc.y + (1.0/2.0)*parentLoc.height, parentLoc.width, (1.0/2.0)*parentLoc.height);
			bottomSearchSpace = cvRect(parentLoc.x, parentLoc.y + (5.0/8.0)*parentLoc.height, parentLoc.width, (3.0/8.0)*parentLoc.height);

			Rect bottomROI(bottomSearchSpace);	//Make a rectangle
			Mat imgBottomROI = Mat(img)(bottomROI);	//Point a cv::Mat header at it (no allocation is done)
			imgBottomROI.copyTo(this->leftEyeBottomTpl);

			cvNamedWindow( "leftEyeBottom", 1 );
			imshow( "leftEyeBottom", this->leftEyeBottomTpl);


			//pink box
			//rectangle(Mat(processedImg),Point(bottomSearchSpace.x,bottomSearchSpace.y),
			//	Point(bottomSearchSpace.x+bottomSearchSpace.width,bottomSearchSpace.y+bottomSearchSpace.height),
			//	CV_RGB(255,0,255),1,0,0);

		}
		else //update search space for NCC
		{
 ////resize sub feature templates and search space based on new face width and height
 //                       //double oldFaceWidth = (double)r.width;
 //                       //double oldFaceHeight = (double)r.height;

 //                       //top
 //                       double oldTopTplWidth = (double)leftEyeTopTpl.cols;
 //                       //double newTopTplWidth = (oldTopTplWidth / oldFaceWidth) * newFaceWidth;
 //                       //double newTopTplWidth = (oldTopTplWidth / this->oldFaceWidth) * newFaceWidth;
 //                       
 //                       //double newTopTplWidth = (oldTopTplWidth / this->oldFaceWidth) * r.width;
 //                       double newTopTplWidth = ceil( (oldTopTplWidth / this->oldFaceWidth) * r.width );

 //                       //prevent against width <= 0
 //                       if(newTopTplWidth < 1)
 //                       {
 //                             newTopTplWidth = 1;
 //                       }

 //                       std::cout << "old face width: " << this->oldFaceWidth << std::endl;
 //                       std::cout << "new face width: " << r.width << std::endl;
 //                       
 //                       std::cout << "oldTopTplWidth: " << oldTopTplWidth << std::endl;
 //                       std::cout << "newTopTplWidth: " << newTopTplWidth << std::endl;


 //                       double oldTopTplHeight = (double)leftEyeTopTpl.rows;
 //                       //double newTopTplHeight = (oldTopTplHeight / oldFaceHeight) * newFaceHeight;
 //                       //double newTopTplHeight = (oldTopTplHeight / this->oldFaceHeight) * newFaceHeight;
 //                       
 //                       //double newTopTplHeight = (oldTopTplHeight / this->oldFaceHeight) * r.height;
 //                       double newTopTplHeight = ceil( (oldTopTplHeight / this->oldFaceHeight) * r.height );
 //                       
 //                       //std::cout << "newTopTplHeight: " << newTopTplHeight << std::endl;

 //                       //prevent against height <= 0
 //                       if(newTopTplHeight < 1)
 //                       {
 //                             newTopTplHeight = 1;
 //                       }

 //                       std::cout << "old face height: " << this->oldFaceHeight << std::endl;
 //                       std::cout << "new face height: " << r.height << std::endl;
 //                       
 //                       std::cout << "oldTopTplHeight: " << oldTopTplHeight << std::endl;
 //                       std::cout << "newTopTplHeight: " << newTopTplHeight << std::endl;

 //                       Mat newLeftEyeTopTpl;
 //                       resize(this->leftEyeTopTpl,newLeftEyeTopTpl,Size((int)newTopTplWidth,(int)newTopTplHeight));

 //                       //swap templates back                 
 //                       //newLeftEyeTopTpl.copyTo(this->leftEyeTopTpl);
 //                       this->leftEyeTopTpl = newLeftEyeTopTpl;


 //                       //bottom
 //                       double oldBottomTplWidth = (double)leftEyeBottomTpl.cols;
 //                       //double newBottomTplWidth = (oldBottomTplWidth / oldFaceWidth) * newFaceWidth;
 //                       //double newBottomTplWidth = (oldBottomTplWidth / this->oldFaceWidth) * newFaceWidth;
 //                       double newBottomTplWidth = (oldBottomTplWidth / this->oldFaceWidth) * r.width;

	//					//std::cout << "newBottomTplWidth: " << newBottomTplWidth << std::endl;

 //                       //prevent against width <= 0
 //                       if(newBottomTplWidth < 1)
 //                       {
 //                             newBottomTplWidth = 1;
 //                       }

 //                       std::cout << "oldBottomTplWidth: " << oldBottomTplWidth << std::endl;
 //                       std::cout << "newBottomTplWidth: " << newBottomTplWidth << std::endl;


 //                       double oldBottomTplHeight = (double)leftEyeBottomTpl.rows;
 //                       //double newBottomTplHeight = (oldBottomTplHeight / oldFaceHeight) * newFaceHeight;
 //                       //double newBottomTplHeight = (oldBottomTplHeight / this->oldFaceHeight) * newFaceHeight;
 //                       double newBottomTplHeight = (oldBottomTplHeight / this->oldFaceHeight) * r.height;
 //                       //std::cout << "newBottomTplHeight: " << newBottomTplHeight << std::endl;

 //                       //prevent against height <= 0
 //                       if(newBottomTplHeight < 1)
 //                       {
 //                             newBottomTplHeight = 1;
 //                       }

 //                       std::cout << "oldBottomTplHeight: " << oldBottomTplHeight << std::endl;
 //                       std::cout << "newBottomTplHeight: " << newBottomTplHeight << std::endl;

 //                       Mat newLeftEyeBottomTpl;
 //                       resize(this->leftEyeBottomTpl,newLeftEyeBottomTpl,Size((int)newBottomTplWidth,(int)newBottomTplHeight));

 //                       //swap templates back                 
 //                       //newLeftEyeBottomTpl.copyTo(this->leftEyeBottomTpl);
 //                       this->leftEyeBottomTpl = newLeftEyeBottomTpl;
 //                       //end resize



			//resize buffers for search space depending on how much face size changed

			int bufferX = (int)( (((double)r.width/(double)this->oldFaceWidth) * this->leftEyeTopLoc.x) - this->leftEyeTopLoc.x );
			//when the face grows smaller, just use no buffer (same search space as previous) to search instead of shrinking the search space
			//b/c the current template may actually be larger than a resized smaller search space
			if(bufferX < 0)
			{
				bufferX = 0;
			}
			
			int bufferY = (int)( (((double)r.height/(double)this->oldFaceHeight) * this->leftEyeTopLoc.y) - this->leftEyeTopLoc.y );
			if(bufferY < 0)
			{
				bufferY = 0;
			}

			//edge cases for top search space
			//use same maximum boundaries as when searching for whole eye template in face
			int topSearchSpaceX = leftEyeTopLoc.x - bufferX;
			if(topSearchSpaceX < this->topLeftPoint.x)
			{
				topSearchSpaceX = this->topLeftPoint.x;
			}

			int topSearchSpaceY = leftEyeTopLoc.y - bufferY;
			if(topSearchSpaceY < (this->topLeftPoint.y + (int)((2.5/8.0)*this->currentFaceHeight)) )
			{
				topSearchSpaceY = (this->topLeftPoint.y + (int)((2.5/8.0)*this->currentFaceHeight));
			}

			int topSearchSpaceWidth = this->leftEyeTopTpl.cols + 2*bufferX;
			if(topSearchSpaceWidth > (this->currentFaceWidth/2))
			{
				topSearchSpaceWidth = (this->currentFaceWidth/2);
			}

			int topSearchSpaceHeight = this->leftEyeTopTpl.rows + 2*bufferY;
			if(topSearchSpaceHeight > ((int)((2.0/8.0)*this->currentFaceHeight)) )
			{
				topSearchSpaceHeight = ((int)((2.0/8.0)*this->currentFaceHeight));
			}

			topSearchSpace = cvRect(topSearchSpaceX, topSearchSpaceY, topSearchSpaceWidth, topSearchSpaceHeight);

			//black rectangle for top search space
			//rectangle(Mat(processedImg),Point(topSearchSpace.x,topSearchSpace.y),
			//	Point(topSearchSpace.x + topSearchSpace.width, topSearchSpace.y + topSearchSpace.height),CV_RGB(0, 0, 0), 1, 0, 0 );


			//std::cout << "top search space: " << topSearchSpace.x << "," << topSearchSpace.y << ": " << topSearchSpace.width << "by" << topSearchSpace.height << std::endl;

			//edge cases for bottom search space
			//use same maximum boundaries as when searching for whole eye template in face
			int bottomSearchSpaceX = leftEyeBottomLoc.x - bufferX;
			if(bottomSearchSpaceX < this->topLeftPoint.x)
			{
				bottomSearchSpaceX = this->topLeftPoint.x;
			}

			int bottomSearchSpaceY = leftEyeBottomLoc.y - bufferY;
			if(bottomSearchSpaceY < (this->topLeftPoint.y + (int)((2.5/8.0)*this->currentFaceHeight)) )
			{
				bottomSearchSpaceY = (this->topLeftPoint.y + (int)((2.5/8.0)*this->currentFaceHeight));
			}

			int bottomSearchSpaceWidth = this->leftEyeBottomTpl.cols + 2*bufferX;
			if(bottomSearchSpaceWidth > (this->currentFaceWidth/2))
			{
				bottomSearchSpaceWidth = (this->currentFaceWidth/2);
			}

			int bottomSearchSpaceHeight = this->leftEyeBottomTpl.rows + 2*bufferY;
			if(bottomSearchSpaceHeight > ((int)((2.0/8.0)*this->currentFaceHeight)) )
			{
				bottomSearchSpaceHeight = ((int)((2.0/8.0)*this->currentFaceHeight));
			}

			bottomSearchSpace = cvRect(bottomSearchSpaceX, bottomSearchSpaceY, bottomSearchSpaceWidth, bottomSearchSpaceHeight);

			//white box
		/*	rectangle(Mat(processedImg),Point(bottomSearchSpace.x,bottomSearchSpace.y),
				Point(bottomSearchSpace.x + bottomSearchSpace.width, bottomSearchSpace.y + bottomSearchSpace.height),CV_RGB(255, 255, 255), 1, 0, 0 );*/

		}//end else


		//update template image by running NCC
		bool topFound = getSearchSpace(img,processedImg,&r,leftEyeTopTpl,topSearchSpace,topLoc);
		bool bottomFound = getSearchSpace(img,processedImg,&r,leftEyeBottomTpl,bottomSearchSpace,bottomLoc);

		//update coordinates because maybe it was only taking the greatest but not surpassing threshold
		//if(topFound && bottomFound)
		if(true)
		{
			this->leftEyeTopLoc.x = topLoc.x;
			this->leftEyeTopLoc.y = topLoc.y;
			this->leftEyeBottomLoc.x = bottomLoc.x;
			this->leftEyeBottomLoc.y = bottomLoc.y;

			//update (crop out) template images
			//referenced from: http://opencv.willowgarage.com/documentation/cpp/c++_cheatsheet.html
			
			//top
			//Rect topROI(topLoc);	//Make a rectangle
			//Mat imgTopROI = Mat(img)(topROI);	//Point a cv::Mat header at it (no allocation is done)
			//imgTopROI.copyTo(this->leftEyeTopTpl);

			cvNamedWindow( "leftEyeTop", 1 );
			imshow( "leftEyeTop", this->leftEyeTopTpl);


			//bottom
			//Rect bottomROI(bottomLoc);	//Make a rectangle
			//Mat imgBottomROI = Mat(img)(bottomROI);	//Point a cv::Mat header at it (no allocation is done)
			//imgBottomROI.copyTo(this->leftEyeBottomTpl);

			cvNamedWindow( "leftEyeBottom", 1 );
			imshow( "leftEyeBottom", this->leftEyeBottomTpl);

			//std::cout << "top bot diff: " << abs(topLoc.y - bottomLoc.y) << std::endl;

		}
		else //clear out templates
		{
			this->leftEyeTopTpl.release();
			this->leftEyeBottomTpl.release();
			//we didnt' find the feature in this frame :(
		}

		std::cout << "top bot diff: " << abs(topLoc.y - bottomLoc.y) << std::endl;


	}//end updateMouthSubFeatureLocs


	/* Update locations of subfeatures by running NCC on them*/
	//void updateEyeSubFeatureLocations(string featureFilename, newFaceCoords, boolean doResize, boolean doCrop, int frame #)
	//void updateEyeSubFeatureLocations(IplImage *img, IplImage *processedImg, double newFaceWidth, double newFaceHeight, CvRect& parentLoc)
	void updateEyeSubFeatureLocations(IplImage *img, IplImage *processedImg, CvRect& r, CvRect& parentLoc)
	{
		//update face with new coordinates
		this->oldFaceWidth = this->currentFaceWidth;
		this->oldFaceHeight = this->currentFaceWidth;

		this->currentFaceWidth = r.width;	//newFaceWidth
		this->currentFaceHeight = r.height;	//newFaceHeight
		
		this->topLeftPoint.x = r.x;
		this->topLeftPoint.y = r.y;


		CvRect topLoc = cvRect(0,0,0,0);			
		CvRect bottomLoc = cvRect(0,0,0,0);

		CvRect topSearchSpace;		// if condition is true, parent feature subsection, else subfeature location with Buffer
		CvRect bottomSearchSpace;

		//subfeature has no previous coordinates, i.e. the template images are empty
		if(leftEyeTopTpl.empty() || leftEyeBottomTpl.empty())
		//if(true)
		{
			//look at parent feature coords for search space
			
			//crop out eye template
			Rect parentROI(parentLoc);	//Make a rectangle
			Mat imgParentROI = Mat(img)(parentROI);	//Point a cv::Mat header at it (no allocation is done)
			//imgParentROI.copyTo(this->leftEyeParentTpl);

			cvNamedWindow( "eyeTop", 1 );
			imshow( "eyeTop", imgParentROI);


			//top subtemplate
			//Mat oldTopTpl = imread("templates/leftEyeTop.jpg",1);			
			//resizeFeatureTemplate(oldTopTpl,r.width,r.height,this->leftEyeTopTpl);
			
			//topSearchSpace = cvRect(parentLoc.x, parentLoc.y,parentLoc.width, parentLoc.height/2);
			topSearchSpace = cvRect(parentLoc.x, parentLoc.y,parentLoc.width, (3.0/8.0)*parentLoc.height);


			Rect topROI(topSearchSpace);
			Mat imgTopROI = Mat(img)(topROI);
			imgTopROI.copyTo(this->leftEyeTopTpl);

			cvNamedWindow( "leftEyeTop", 1 );
			imshow( "leftEyeTop", this->leftEyeTopTpl);

			
			//yellow box
			//rectangle(Mat(processedImg),Point(topSearchSpace.x,topSearchSpace.y),
			//	Point(topSearchSpace.x+topSearchSpace.width,topSearchSpace.y+topSearchSpace.height),
			//	CV_RGB(255,255,0),1,0,0);


			//bottom subtemplate
			//Mat oldBottomTpl = imread("templates/leftEyeBottom.jpg",1);
			//resizeFeatureTemplate(oldBottomTpl,r.width,r.height,this->leftEyeBottomTpl);

			//bottomSearchSpace = cvRect(parentLoc.x, parentLoc.y + (1.0/2.0)*parentLoc.height, parentLoc.width, (1.0/2.0)*parentLoc.height);
			bottomSearchSpace = cvRect(parentLoc.x, parentLoc.y + (5.0/8.0)*parentLoc.height, parentLoc.width, (3.0/8.0)*parentLoc.height);

			Rect bottomROI(bottomSearchSpace);	//Make a rectangle
			Mat imgBottomROI = Mat(img)(bottomROI);	//Point a cv::Mat header at it (no allocation is done)
			imgBottomROI.copyTo(this->leftEyeBottomTpl);

			cvNamedWindow( "leftEyeBottom", 1 );
			imshow( "leftEyeBottom", this->leftEyeBottomTpl);


			//pink box
			//rectangle(Mat(processedImg),Point(bottomSearchSpace.x,bottomSearchSpace.y),
			//	Point(bottomSearchSpace.x+bottomSearchSpace.width,bottomSearchSpace.y+bottomSearchSpace.height),
			//	CV_RGB(255,0,255),1,0,0);

		}
		else //update search space for NCC
		{
 ////resize sub feature templates and search space based on new face width and height
 //                       //double oldFaceWidth = (double)r.width;
 //                       //double oldFaceHeight = (double)r.height;

 //                       //top
 //                       double oldTopTplWidth = (double)leftEyeTopTpl.cols;
 //                       //double newTopTplWidth = (oldTopTplWidth / oldFaceWidth) * newFaceWidth;
 //                       //double newTopTplWidth = (oldTopTplWidth / this->oldFaceWidth) * newFaceWidth;
 //                       
 //                       //double newTopTplWidth = (oldTopTplWidth / this->oldFaceWidth) * r.width;
 //                       double newTopTplWidth = ceil( (oldTopTplWidth / this->oldFaceWidth) * r.width );

 //                       //prevent against width <= 0
 //                       if(newTopTplWidth < 1)
 //                       {
 //                             newTopTplWidth = 1;
 //                       }

 //                       std::cout << "old face width: " << this->oldFaceWidth << std::endl;
 //                       std::cout << "new face width: " << r.width << std::endl;
 //                       
 //                       std::cout << "oldTopTplWidth: " << oldTopTplWidth << std::endl;
 //                       std::cout << "newTopTplWidth: " << newTopTplWidth << std::endl;


 //                       double oldTopTplHeight = (double)leftEyeTopTpl.rows;
 //                       //double newTopTplHeight = (oldTopTplHeight / oldFaceHeight) * newFaceHeight;
 //                       //double newTopTplHeight = (oldTopTplHeight / this->oldFaceHeight) * newFaceHeight;
 //                       
 //                       //double newTopTplHeight = (oldTopTplHeight / this->oldFaceHeight) * r.height;
 //                       double newTopTplHeight = ceil( (oldTopTplHeight / this->oldFaceHeight) * r.height );
 //                       
 //                       //std::cout << "newTopTplHeight: " << newTopTplHeight << std::endl;

 //                       //prevent against height <= 0
 //                       if(newTopTplHeight < 1)
 //                       {
 //                             newTopTplHeight = 1;
 //                       }

 //                       std::cout << "old face height: " << this->oldFaceHeight << std::endl;
 //                       std::cout << "new face height: " << r.height << std::endl;
 //                       
 //                       std::cout << "oldTopTplHeight: " << oldTopTplHeight << std::endl;
 //                       std::cout << "newTopTplHeight: " << newTopTplHeight << std::endl;

 //                       Mat newLeftEyeTopTpl;
 //                       resize(this->leftEyeTopTpl,newLeftEyeTopTpl,Size((int)newTopTplWidth,(int)newTopTplHeight));

 //                       //swap templates back                 
 //                       //newLeftEyeTopTpl.copyTo(this->leftEyeTopTpl);
 //                       this->leftEyeTopTpl = newLeftEyeTopTpl;


 //                       //bottom
 //                       double oldBottomTplWidth = (double)leftEyeBottomTpl.cols;
 //                       //double newBottomTplWidth = (oldBottomTplWidth / oldFaceWidth) * newFaceWidth;
 //                       //double newBottomTplWidth = (oldBottomTplWidth / this->oldFaceWidth) * newFaceWidth;
 //                       double newBottomTplWidth = (oldBottomTplWidth / this->oldFaceWidth) * r.width;

	//					//std::cout << "newBottomTplWidth: " << newBottomTplWidth << std::endl;

 //                       //prevent against width <= 0
 //                       if(newBottomTplWidth < 1)
 //                       {
 //                             newBottomTplWidth = 1;
 //                       }

 //                       std::cout << "oldBottomTplWidth: " << oldBottomTplWidth << std::endl;
 //                       std::cout << "newBottomTplWidth: " << newBottomTplWidth << std::endl;


 //                       double oldBottomTplHeight = (double)leftEyeBottomTpl.rows;
 //                       //double newBottomTplHeight = (oldBottomTplHeight / oldFaceHeight) * newFaceHeight;
 //                       //double newBottomTplHeight = (oldBottomTplHeight / this->oldFaceHeight) * newFaceHeight;
 //                       double newBottomTplHeight = (oldBottomTplHeight / this->oldFaceHeight) * r.height;
 //                       //std::cout << "newBottomTplHeight: " << newBottomTplHeight << std::endl;

 //                       //prevent against height <= 0
 //                       if(newBottomTplHeight < 1)
 //                       {
 //                             newBottomTplHeight = 1;
 //                       }

 //                       std::cout << "oldBottomTplHeight: " << oldBottomTplHeight << std::endl;
 //                       std::cout << "newBottomTplHeight: " << newBottomTplHeight << std::endl;

 //                       Mat newLeftEyeBottomTpl;
 //                       resize(this->leftEyeBottomTpl,newLeftEyeBottomTpl,Size((int)newBottomTplWidth,(int)newBottomTplHeight));

 //                       //swap templates back                 
 //                       //newLeftEyeBottomTpl.copyTo(this->leftEyeBottomTpl);
 //                       this->leftEyeBottomTpl = newLeftEyeBottomTpl;
 //                       //end resize



			//resize buffers for search space depending on how much face size changed

			int bufferX = (int)( (((double)r.width/(double)this->oldFaceWidth) * this->leftEyeTopLoc.x) - this->leftEyeTopLoc.x );
			//when the face grows smaller, just use no buffer (same search space as previous) to search instead of shrinking the search space
			//b/c the current template may actually be larger than a resized smaller search space
			if(bufferX < 0)
			{
				bufferX = 0;
			}
			
			int bufferY = (int)( (((double)r.height/(double)this->oldFaceHeight) * this->leftEyeTopLoc.y) - this->leftEyeTopLoc.y );
			if(bufferY < 0)
			{
				bufferY = 0;
			}

			//edge cases for top search space
			//use same maximum boundaries as when searching for whole eye template in face
			int topSearchSpaceX = leftEyeTopLoc.x - bufferX;
			if(topSearchSpaceX < this->topLeftPoint.x)
			{
				topSearchSpaceX = this->topLeftPoint.x;
			}

			int topSearchSpaceY = leftEyeTopLoc.y - bufferY;
			if(topSearchSpaceY < (this->topLeftPoint.y + (int)((2.5/8.0)*this->currentFaceHeight)) )
			{
				topSearchSpaceY = (this->topLeftPoint.y + (int)((2.5/8.0)*this->currentFaceHeight));
			}

			int topSearchSpaceWidth = this->leftEyeTopTpl.cols + 2*bufferX;
			if(topSearchSpaceWidth > (this->currentFaceWidth/2))
			{
				topSearchSpaceWidth = (this->currentFaceWidth/2);
			}

			int topSearchSpaceHeight = this->leftEyeTopTpl.rows + 2*bufferY;
			if(topSearchSpaceHeight > ((int)((2.0/8.0)*this->currentFaceHeight)) )
			{
				topSearchSpaceHeight = ((int)((2.0/8.0)*this->currentFaceHeight));
			}

			topSearchSpace = cvRect(topSearchSpaceX, topSearchSpaceY, topSearchSpaceWidth, topSearchSpaceHeight);

			//black rectangle for top search space
			//rectangle(Mat(processedImg),Point(topSearchSpace.x,topSearchSpace.y),
			//	Point(topSearchSpace.x + topSearchSpace.width, topSearchSpace.y + topSearchSpace.height),CV_RGB(0, 0, 0), 1, 0, 0 );


			//std::cout << "top search space: " << topSearchSpace.x << "," << topSearchSpace.y << ": " << topSearchSpace.width << "by" << topSearchSpace.height << std::endl;

			//edge cases for bottom search space
			//use same maximum boundaries as when searching for whole eye template in face
			int bottomSearchSpaceX = leftEyeBottomLoc.x - bufferX;
			if(bottomSearchSpaceX < this->topLeftPoint.x)
			{
				bottomSearchSpaceX = this->topLeftPoint.x;
			}

			int bottomSearchSpaceY = leftEyeBottomLoc.y - bufferY;
			if(bottomSearchSpaceY < (this->topLeftPoint.y + (int)((2.5/8.0)*this->currentFaceHeight)) )
			{
				bottomSearchSpaceY = (this->topLeftPoint.y + (int)((2.5/8.0)*this->currentFaceHeight));
			}

			int bottomSearchSpaceWidth = this->leftEyeBottomTpl.cols + 2*bufferX;
			if(bottomSearchSpaceWidth > (this->currentFaceWidth/2))
			{
				bottomSearchSpaceWidth = (this->currentFaceWidth/2);
			}

			int bottomSearchSpaceHeight = this->leftEyeBottomTpl.rows + 2*bufferY;
			if(bottomSearchSpaceHeight > ((int)((2.0/8.0)*this->currentFaceHeight)) )
			{
				bottomSearchSpaceHeight = ((int)((2.0/8.0)*this->currentFaceHeight));
			}

			bottomSearchSpace = cvRect(bottomSearchSpaceX, bottomSearchSpaceY, bottomSearchSpaceWidth, bottomSearchSpaceHeight);

			//white box
		/*	rectangle(Mat(processedImg),Point(bottomSearchSpace.x,bottomSearchSpace.y),
				Point(bottomSearchSpace.x + bottomSearchSpace.width, bottomSearchSpace.y + bottomSearchSpace.height),CV_RGB(255, 255, 255), 1, 0, 0 );*/

		}//end else


		//update template image by running NCC
		bool topFound = getSearchSpace(img,processedImg,&r,leftEyeTopTpl,topSearchSpace,topLoc);
		bool bottomFound = getSearchSpace(img,processedImg,&r,leftEyeBottomTpl,bottomSearchSpace,bottomLoc);

		//update coordinates because maybe it was only taking the greatest but not surpassing threshold
		//if(topFound && bottomFound)
		if(true)
		{
			this->leftEyeTopLoc.x = topLoc.x;
			this->leftEyeTopLoc.y = topLoc.y;
			this->leftEyeBottomLoc.x = bottomLoc.x;
			this->leftEyeBottomLoc.y = bottomLoc.y;

			//update (crop out) template images
			//referenced from: http://opencv.willowgarage.com/documentation/cpp/c++_cheatsheet.html
			
			//top
			//Rect topROI(topLoc);	//Make a rectangle
			//Mat imgTopROI = Mat(img)(topROI);	//Point a cv::Mat header at it (no allocation is done)
			//imgTopROI.copyTo(this->leftEyeTopTpl);

			cvNamedWindow( "leftEyeTop", 1 );
			imshow( "leftEyeTop", this->leftEyeTopTpl);


			//bottom
			//Rect bottomROI(bottomLoc);	//Make a rectangle
			//Mat imgBottomROI = Mat(img)(bottomROI);	//Point a cv::Mat header at it (no allocation is done)
			//imgBottomROI.copyTo(this->leftEyeBottomTpl);

			cvNamedWindow( "leftEyeBottom", 1 );
			imshow( "leftEyeBottom", this->leftEyeBottomTpl);

			//std::cout << "top bot diff: " << abs(topLoc.y - bottomLoc.y) << std::endl;

		}
		else //clear out templates
		{
			this->leftEyeTopTpl.release();
			this->leftEyeBottomTpl.release();
			//we didnt' find the feature in this frame :(
		}

		std::cout << "top bot diff: " << abs(topLoc.y - bottomLoc.y) << std::endl;


	}//end updateEyeSubFeatureLocations


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
		CvRect leftEyeLoc;
		bool leftEyeFound = getSearchSpace(img,processedImg,r,leftEyeTpl,leftEyeSearchSpace,leftEyeLoc);
		

		if(!leftEyeFound)
		{
			result = false;
		}
		else
		{
			//resize each parent feature (except left eye)      
			//each parent feature has specific search space (area of face)
			//run NCC on parent feature (except left eye) to get search space for sub features
			
			//void updateEyeSubFeatureLocations(IplImage *img, IplImage *processedImg, double newFaceWidth, double newFaceHeight, CvRect& parentLoc)
			//updateEyeSubFeatureLocations(img, processedImg, r->width, r->height, leftEyeLoc);
			
			//uncomment below:
			//updateEyeSubFeatureLocations(img, processedImg, *r, leftEyeLoc);

			////left eye left
			//Mat oldLeftEyeLeftTpl = imread("templates/leftEyeLeft.jpg",1);
			//Mat leftEyeLeftTpl;
			//resizeFeatureTemplate(oldLeftEyeLeftTpl,newFaceWidth,newFaceHeight,leftEyeLeftTpl);
			//CvRect leftEyeLeftSearchSpace = cvRect(leftEyeLoc.x,leftEyeLoc.y,
			//	leftEyeLoc.width/2,leftEyeLoc.height);
			//CvRect leftEyeLeftSubSearchSpace;
			//bool leftEyeLeftFound = getSearchSpace(img,processedImg,r,leftEyeLeftTpl,leftEyeLeftSearchSpace,leftEyeLeftSubSearchSpace);

			////left eye right
			//Mat oldLeftEyeRightTpl = imread("templates/leftEyeRight.jpg",1);
			//Mat leftEyeRightTpl;
			//resizeFeatureTemplate(oldLeftEyeRightTpl,newFaceWidth,newFaceHeight,leftEyeRightTpl);	
			//CvRect leftEyeRightSearchSpace = cvRect((leftEyeLoc.x + leftEyeLoc.width/2),leftEyeLoc.y,
			//	leftEyeLoc.width/2,leftEyeLoc.height);
			//CvRect leftEyeRightSubSearchSpace;
			//bool leftEyeRightFound = getSearchSpace(img,processedImg,r,leftEyeRightTpl,leftEyeRightSearchSpace,leftEyeRightSubSearchSpace);

			

			////left eye top
			//Mat oldLeftEyeTopTpl = imread("templates/leftEyeTop.jpg",1);
			//Mat leftEyeTopTpl;
			//resizeFeatureTemplate(oldLeftEyeTopTpl,newFaceWidth,newFaceHeight,leftEyeTopTpl);
			//CvRect leftEyeTopSearchSpace = cvRect((leftEyeLoc.x),(leftEyeLoc.y),
			//	leftEyeLoc.width,leftEyeLoc.height/2);
			//CvRect leftEyeTopSubSearchSpace;
			//bool leftEyeTopFound = getSearchSpace(img,processedImg,r,leftEyeTopTpl,leftEyeTopSearchSpace,leftEyeTopSubSearchSpace);

			////left eye bottom
			//Mat oldLeftEyeBottomTpl = imread("templates/leftEyeBottom.jpg",1);
			//Mat leftEyeBottomTpl;
			//resizeFeatureTemplate(oldLeftEyeBottomTpl,newFaceWidth,newFaceHeight,leftEyeBottomTpl);
			//CvRect leftEyeBottomSearchSpace = cvRect((leftEyeLoc.x),(leftEyeLoc.y + leftEyeLoc.height/2),
			//	leftEyeLoc.width,leftEyeLoc.height/2);
			//CvRect leftEyeBottomSubSearchSpace;
			//bool leftEyeBottomFound = getSearchSpace(img,processedImg,r,leftEyeBottomTpl,leftEyeBottomSearchSpace,leftEyeBottomSubSearchSpace);


			//uncomment starting here:

			////right eye
			//Mat oldRightEyeTpl = imread("templates/rightEye.jpg",1);
			//Mat rightEyeTpl;
			//resizeFeatureTemplate(oldRightEyeTpl,newFaceWidth,newFaceHeight,rightEyeTpl);
			////CvRect rightEyeSearchSpace = cvRect((r->x + r->width/2), (r->y + r->height/4), r->width/2, (int)((3.0/8.0)*r->height));
			////CvRect rightEyeSearchSpace = cvRect((r->x + r->width/2), (r->y + (int)((2.5/8.0)*r->height)), r->width/2, (int)((2.5/8.0)*r->height));
			//CvRect rightEyeSearchSpace = cvRect((r->x + r->width/2), (r->y + (int)((2.5/8.0)*r->height)), r->width/2, (int)((2.0/8.0)*r->height));
			//CvRect rightEyeSubSearchSpace;
			//bool rightEyeFound = getSearchSpace(img,processedImg,r,rightEyeTpl,rightEyeSearchSpace,rightEyeSubSearchSpace);

			////left eyebrow
			//Mat oldLeftEyebrowTpl = imread("templates/leftEyebrow.jpg",1);
			//Mat leftEyebrowTpl;
			//resizeFeatureTemplate(oldLeftEyebrowTpl,newFaceWidth,newFaceHeight,leftEyebrowTpl);
			//CvRect leftEyebrowSearchSpace = cvRect(r->x, r->y, r->width/2, (int)((3.0/8.0)*r->height));
			////CvRect leftEyebrowSearchSpace = cvRect(r->x, r->y, r->width/2, (int)((4.0/8.0)*r->height));
			//CvRect leftEyebrowSubSearchSpace;
			//bool leftEyebrowFound = getSearchSpace(img,processedImg,r,leftEyebrowTpl,leftEyebrowSearchSpace,leftEyebrowSubSearchSpace);

			////right eyebrow
			//Mat oldRightEyebrowTpl = imread("templates/rightEyebrow.jpg",1);
			//Mat rightEyebrowTpl;
			//resizeFeatureTemplate(oldRightEyebrowTpl,newFaceWidth,newFaceHeight,rightEyebrowTpl);
			//CvRect rightEyebrowSearchSpace = cvRect((r->x + r->width/2),r->y, r->width/2, (int)((3.0/8.0)*r->height));
			//CvRect rightEyebrowSubSearchSpace;
			//bool rightEyebrowFound = getSearchSpace(img,processedImg,r,rightEyebrowTpl,rightEyebrowSearchSpace,rightEyebrowSubSearchSpace);

			//mouth
			Mat oldMouthTpl = imread("templates/mouth.jpg",1);
			Mat mouthTpl;
			resizeFeatureTemplate(oldMouthTpl,newFaceWidth,newFaceHeight,mouthTpl);
			CvRect mouthSearchSpace = cvRect(r->x + (int)((2.0/8.0)*r->x), (r->y +  (int)((5.0/8.0)*r->height)), (int)((4.0/8.0)*r->width), (int)((3.0/8.0)*r->height));
			CvRect mouthLoc;
			bool mouthFound = getSearchSpace(img,processedImg,r,mouthTpl,mouthSearchSpace,mouthLoc);
			
			updateMouthSubFeatureLocs(img, processedImg, *r, mouthLoc);


			result = true;
		}

		return result;
	}
	
};

#endif
