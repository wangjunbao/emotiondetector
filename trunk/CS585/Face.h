//#pragma once
#ifndef FACE
#define FACE
#include <cv.h>
//#include <math.h>
using namespace cv;

class Face
{
private:
	//face coordinates and dimensions:
	Point topLeftPoint;
	int oldFaceWidth;
	int oldFaceHeight;
	int currentFaceWidth;
	int currentFaceHeight;



	//mouth
	Mat mouthTpl;
	CvRect mouthLoc;

	Mat mouthTopTpl;
	Point mouthTopLoc;
	Mat mouthBottomTpl;
	Point mouthBottomLoc;

	//left eyebrow
	Mat leftEyebrowTpl;
	CvRect leftEyebrowLoc;

	//right eyebrow
	Mat rightEyebrowTpl;
	CvRect rightEyebrowLoc;

	//left eye
	Mat leftEyeTpl;
	CvRect leftEyeLoc;

	Mat leftEye;
	Point leftEyeTopLoc;
	Point leftEyeBottomLoc;
	Mat leftEyeTopTpl;
	Mat leftEyeBottomTpl;

	//right eye
	Mat rightEyeTpl;
	CvRect rightEyeLoc;


public:
	Face(CvRect r) 
	{
		this->topLeftPoint.x = r.x;
		this->topLeftPoint.y = r.y;

		this->currentFaceWidth = r.width;
		this->currentFaceHeight = r.height;
	}
	
	Point getTopLeftPoint()
	{
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
	
	/* Resize feature templates based on new dimensions of the face */
	void resizeFeatureTemplate(Mat& oldFeatureTemplateImg, double newFaceWidth, double newFaceHeight, Mat &resizedFeatureImg )
	{	
		double oldFaceWidth = 196;	//Average face	//228 //Howell's face;
		double oldFaceHeight = 196;	//Average face	//228 //Howell's face;

		double oldFeatureWidth = (double)oldFeatureTemplateImg.cols;
		double oldFeatureHeight = (double)oldFeatureTemplateImg.rows;
			
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

		//left eyebrow
		namedWindow( "left eyebrow", 1 );
		imshow("left eyebrow",this->leftEyebrowTpl);

		//right eyebrow
		namedWindow( "right eyebrow", 1 );
		imshow("right eyebrow",this->rightEyebrowTpl);

		//left eye
		namedWindow( "left eye", 1 );
		imshow("left eye",this->leftEyeTpl);

		//right eye
		namedWindow( "right eye", 1 );
		imshow("right eye",this->rightEyeTpl);

		//mouth
		namedWindow( "mouth", 1 );
		imshow("mouth",this->mouthTpl);
	}


	/* 
		Run NCC on a template to get the search space for sub templates 
		Return true if NCC was above a threshold, otherwise false
	*/
	bool getSearchSpace(IplImage *img, IplImage *processedImg, 
		CvRect *r, Mat& tpl, CvRect& inputSearchSpace, CvRect& outputSearchSpace, bool detailedOutput=false)
	{
		bool result = false;
		double THRESH = 0.50;

		//Ref: http://nashruddin.com/OpenCV_Region_of_Interest_(ROI)
		
		//draw a box around the search space
		//blue box
		if(detailedOutput == true)
		{
			rectangle(Mat(processedImg),Point(inputSearchSpace.x,inputSearchSpace.y),Point(inputSearchSpace.x+inputSearchSpace.width, inputSearchSpace.y+inputSearchSpace.height),CV_RGB(0, 0, 255), 1, 0, 0 );
		}

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

		//draw a box around the found feature
		//green box
		if(detailedOutput == true)
		{
			rectangle(Mat(processedImg),maxloc,Point(maxloc.x + tpl.cols, maxloc.y + tpl.rows),CV_RGB(0, 255, 0), 1, 0, 0 );
		}

		outputSearchSpace = cvRect(inputSearchSpace.x + maxloc.x, inputSearchSpace.y + maxloc.y, tpl.cols, tpl.rows);
		//std::cout << "**************outputSearchSpace: " << outputSearchSpace.width << " by " << outputSearchSpace.height << std::endl;

		cvResetImageROI(img);
		cvResetImageROI(processedImg);

		result = maxval > THRESH;

		//for debugging:
		//draw a red box around the face if the feature was found
		//if(result == true)
		//{
		//	//red box
		//	cvRectangle( processedImg,
		//		cvPoint( r->x, r->y ),
		//		cvPoint( r->x + r->width, r->y + r->height ),
		//		CV_RGB( 255, 0, 0 ), 1, 8, 0 );
		//}

		//print out search space relative to entire image, not just ROI
		if(detailedOutput == true)
		{
			//std::cout << "max: " << "(" << inputSearchSpace.x + maxloc.x << "," << inputSearchSpace.y + maxloc.y << "): " << maxval << std::endl;
		}

		return result;

	}

	/* update face with new coordinates */
	void updateFaceCoords(CvRect *r)
	{
		this->oldFaceWidth = this->currentFaceWidth;
		this->oldFaceHeight = this->currentFaceWidth;
		this->currentFaceWidth = r->width;
		this->currentFaceHeight = r->height;
		this->topLeftPoint.x = r->x;
		this->topLeftPoint.y = r->y;
	}


	/* 
		Update locations of mouth subfeatures
		Return true if update successful, false if unsuccessful (ie: features were lost)
	*/
	bool updateMouthSubFeatureLocs(IplImage *img, IplImage *processedImg, CvRect *r)
	{
		//initialize search spaces to 0's
		CvRect topLoc = cvRect(0,0,0,0);			
		CvRect bottomLoc = cvRect(0,0,0,0);

		CvRect topSearchSpace;		// if condition is true, parent feature subsection, else subfeature location with Buffer
		CvRect bottomSearchSpace;

		//subfeature has no previous coordinates, i.e. the template images are empty
		if(mouthTopTpl.empty() || mouthBottomTpl.empty())
		{
			//look at parent feature coords for search space
			
			//crop out mouth template

			//should run ncc on parent here because we never do!

			int newFaceWidth = r->width;
			int newFaceHeight = r->height;

			//mouth
			Mat oldMouthTpl = imread("templates/mouth.jpg",1); //just to get the dimensions
			//Mat oldMouthTpl = this->mouthTpl;
			Mat mouthTpl;
			resizeFeatureTemplate(oldMouthTpl,newFaceWidth,newFaceHeight,mouthTpl);
			
			CvRect mouthSearchSpace = cvRect(r->x + (int)((2.0/8.0)*r->width), 
				(r->y +  (int)((5.0/8.0)*r->height)), (int)((4.0/8.0)*r->width), (int)((3.0/8.0)*r->height));
			
			CvRect mouthLoc;
			bool mouthFound = getSearchSpace(img,processedImg,r,mouthTpl,mouthSearchSpace,mouthLoc,true);

			std::cout << "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~MOUTH FOUND: " << mouthFound << std::endl;

			if(mouthFound == false)
			{
				return false;
			}


			//CvRect parentLoc = this->mouthLoc;
			CvRect parentLoc = mouthLoc;

			Rect parentROI(parentLoc);	//Make a rectangle
			//Rect parentROI(this->mouthLoc);
			Mat imgParentROI = Mat(img)(parentROI);	//Point a cv::Mat header at it (no allocation is done)
			imgParentROI.copyTo(this->mouthTpl);

			cvNamedWindow( "mouth", 1 );
			//imshow( "mouthTop", imgParentROI);
			imshow("mouth",this->mouthTpl);

			
			//std::cout << "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~MOUTH FOUND: " << mouthFound << std::endl;

			//if(mouthFound == false)
			//{
			//	std::cout << "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~FACE LOST" << std::endl;
			//	return false;
			//}



			//top subtemplate
			//Mat oldTopTpl = imread("templates/mouthTop.jpg",1);			
			//resizeFeatureTemplate(oldTopTpl,r.width,r.height,this->mouthTopTpl);
			//topSearchSpace = cvRect(parentLoc.x, parentLoc.y,parentLoc.width, parentLoc.height/2);
			
			topSearchSpace = cvRect(parentLoc.x + (2.0/8.0)*parentLoc.width, parentLoc.y,
				(4.0/8.0)*parentLoc.width, (3.0/8.0)*parentLoc.height);			
			Rect topROI(topSearchSpace);
			Mat imgTopROI = Mat(img)(topROI);
			imgTopROI.copyTo(this->mouthTopTpl);

			//maybe we should run the ncc here instead of just doing location search

			

			cvNamedWindow( "mouthTop", 1 );
			imshow( "mouthTop", this->mouthTopTpl);
			

			
			//yellow box
			rectangle(Mat(processedImg),Point(topSearchSpace.x,topSearchSpace.y),
				Point(topSearchSpace.x+topSearchSpace.width,topSearchSpace.y+topSearchSpace.height),
				CV_RGB(255,255,0),1,0,0);


			//bottom subtemplate
	/*		Mat oldBottomTpl = imread("templates/mouthBottom.jpg",1);
			resizeFeatureTemplate(oldBottomTpl,r.width,r.height,this->mouthBottomTpl);
			bottomSearchSpace = cvRect(parentLoc.x, parentLoc.y + (1.0/2.0)*parentLoc.height, parentLoc.width, (1.0/2.0)*parentLoc.height);
			*/
			
			bottomSearchSpace = cvRect(parentLoc.x + (2.0/8.0)*parentLoc.width, parentLoc.y + (5.0/8.0)*parentLoc.height, 
				(4.0/8.0)*parentLoc.width, (3.0/8.0)*parentLoc.height);
			Rect bottomROI(bottomSearchSpace);	//Make a rectangle
			Mat imgBottomROI = Mat(img)(bottomROI);	//Point a cv::Mat header at it (no allocation is done)
			imgBottomROI.copyTo(this->mouthBottomTpl);
			
			//maybe we should run the ncc here instead of just doing location search

			cvNamedWindow( "mouthBottom", 1 );
			imshow( "mouthBottom", this->mouthBottomTpl);
			


			//pink box
			rectangle(Mat(processedImg),Point(bottomSearchSpace.x,bottomSearchSpace.y),
				Point(bottomSearchSpace.x+bottomSearchSpace.width,bottomSearchSpace.y+bottomSearchSpace.height),
				CV_RGB(255,0,255),1,0,0);

		}
		else //update search space for NCC
		{




			//resize buffers for search space depending on how much face size changed

			//int bufferX = (int)( (((double)r.width/(double)this->oldFaceWidth) * this->mouthTopLoc.x) - this->mouthTopLoc.x );
			
			/*int bufferX = this->mouthTopTpl.cols + 
				(int)( (((double)r.width/(double)this->oldFaceWidth) * this->mouthTopLoc.x) - this->mouthTopLoc.x );
			*/
int bufferX = 15;//10;//5;
			//std::cout << "top width: " << this->mouthTopTpl.cols << std::endl;
			//std::cout << "buffer x: " << bufferX << std::endl;
			


//when the face grows smaller, just use no buffer (same search space as previous) to search instead of shrinking the search space
			//b/c the current template may actually be larger than a resized smaller search space
			if(bufferX < 0)
			{
				bufferX = 0;
			}
			
			/*int bufferY = this->mouthTopTpl.rows + 
				(int)( (((double)r.height/(double)this->oldFaceHeight) * this->mouthTopLoc.y) - this->mouthTopLoc.y );
			*/
			int bufferY = 15;//10;//5;
			//std::cout << "buffer y: " << bufferY << std::endl;
			
			if(bufferY < 0)
			{
				bufferY = 0;
			}

			//edge cases for top search space
			//use same maximum boundaries as when searching for whole mouth template in face

			CvRect mouthSearchSpace = cvRect(this->topLeftPoint.x + (int)((2.0/8.0)*this->topLeftPoint.x), 
				(this->topLeftPoint.y +  (int)((5.0/8.0)*this->currentFaceHeight)), 
				(int)((4.0/8.0)*this->currentFaceWidth), 
				(int)((3.0/8.0)*this->currentFaceHeight));
			//it is possible for the bottom lip to leave the "face" when the mouth is opened really wide

			int topSearchSpaceX = mouthTopLoc.x - bufferX;
			//if(topSearchSpaceX < this->topLeftPoint.x)
			if(topSearchSpaceX < mouthSearchSpace.x)
			{
				topSearchSpaceX = mouthSearchSpace.x;
			}

			int topSearchSpaceY = mouthTopLoc.y - bufferY;
			//if(topSearchSpaceY < (this->topLeftPoint.y + (int)((2.5/8.0)*this->currentFaceHeight)) )
			if(topSearchSpaceY < mouthSearchSpace.y)
			{
				//topSearchSpaceY = (this->topLeftPoint.y + (int)((2.5/8.0)*this->currentFaceHeight));
				topSearchSpaceY = mouthSearchSpace.y;
			}

			int topSearchSpaceWidth = this->mouthTopTpl.cols + 2*bufferX;
			//if(topSearchSpaceWidth > (this->currentFaceWidth/2))
			if(topSearchSpaceWidth > mouthSearchSpace.width)
			{
				//topSearchSpaceWidth = (this->currentFaceWidth/2);
				topSearchSpaceWidth = mouthSearchSpace.width;
			}

			int topSearchSpaceHeight = this->mouthTopTpl.rows + 2*bufferY;
			//if(topSearchSpaceHeight > ((int)((2.0/8.0)*this->currentFaceHeight)) )
			if(topSearchSpaceHeight > mouthSearchSpace.height)
			{
				//topSearchSpaceHeight = ((int)((2.0/8.0)*this->currentFaceHeight));
				topSearchSpaceHeight = mouthSearchSpace.height;

			}

			topSearchSpace = cvRect(topSearchSpaceX, topSearchSpaceY, topSearchSpaceWidth, topSearchSpaceHeight);

			//black rectangle for top search space
			rectangle(Mat(processedImg),Point(topSearchSpace.x,topSearchSpace.y),
				Point(topSearchSpace.x + topSearchSpace.width, topSearchSpace.y + topSearchSpace.height),CV_RGB(0, 0, 0), 1, 0, 0 );


			//std::cout << "top search space: " << topSearchSpace.x << "," << topSearchSpace.y << ": " << topSearchSpace.width << "by" << topSearchSpace.height << std::endl;


				//		CvRect mouthSearchSpace = cvRect(this->topLeftPoint.x + (int)((2.0/8.0)*this->topLeftPoint.x), 
				//(this->topLeftPoint.y +  (int)((5.0/8.0)*this->currentFaceHeight)), 
				//(int)((4.0/8.0)*this->currentFaceWidth), 
				//(int)((3.0/8.0)*this->currentFaceHeight));

			//edge cases for bottom search space
			//use same maximum boundaries as when searching for whole mouth template in face
			int bottomSearchSpaceX = mouthBottomLoc.x - bufferX;
			//if(bottomSearchSpaceX < this->topLeftPoint.x)
			if(bottomSearchSpaceX < mouthSearchSpace.x)
			{
				//bottomSearchSpaceX = this->topLeftPoint.x;
				bottomSearchSpaceX = mouthSearchSpace.x;
			}

			int bottomSearchSpaceY = mouthBottomLoc.y - bufferY;
			//if(bottomSearchSpaceY < (this->topLeftPoint.y + (int)((2.5/8.0)*this->currentFaceHeight)) )
			if(bottomSearchSpaceY < mouthSearchSpace.y)
			{
				//bottomSearchSpaceY = (this->topLeftPoint.y + (int)((2.5/8.0)*this->currentFaceHeight));
				bottomSearchSpaceY = mouthSearchSpace.y;
			}

			int bottomSearchSpaceWidth = this->mouthBottomTpl.cols + 2*bufferX;
			//if(bottomSearchSpaceWidth > (this->currentFaceWidth/2))
			if(bottomSearchSpaceWidth > mouthSearchSpace.width)
			{
				//bottomSearchSpaceWidth = (this->currentFaceWidth/2);
				bottomSearchSpaceWidth = mouthSearchSpace.width;
			}

			int bottomSearchSpaceHeight = this->mouthBottomTpl.rows + 2*bufferY;
			//if(bottomSearchSpaceHeight > ((int)((2.0/8.0)*this->currentFaceHeight)) )

			//possible for lower lip to extend outside of the "face"
			if(bottomSearchSpaceHeight > mouthSearchSpace.height)
			{
				//bottomSearchSpaceHeight = ((int)((2.0/8.0)*this->currentFaceHeight));
				bottomSearchSpaceHeight = mouthSearchSpace.height;
			}

			bottomSearchSpace = cvRect(bottomSearchSpaceX, bottomSearchSpaceY, bottomSearchSpaceWidth, bottomSearchSpaceHeight);

			//white box
			rectangle(Mat(processedImg),Point(bottomSearchSpace.x,bottomSearchSpace.y),
				Point(bottomSearchSpace.x + bottomSearchSpace.width, bottomSearchSpace.y + bottomSearchSpace.height),CV_RGB(255, 255, 255), 1, 0, 0 );

		}//end else


		//update template image by running NCC
		bool topFound = getSearchSpace(img,processedImg,r,mouthTopTpl,topSearchSpace,topLoc,true);
		bool bottomFound = getSearchSpace(img,processedImg,r,mouthBottomTpl,bottomSearchSpace,bottomLoc,true);

		//update coordinates because maybe it was only taking the greatest but not surpassing threshold
		if(topFound && bottomFound)
		//if(true)
		{
			this->mouthTopLoc.x = topLoc.x;
			this->mouthTopLoc.y = topLoc.y;
			this->mouthBottomLoc.x = bottomLoc.x;
			this->mouthBottomLoc.y = bottomLoc.y;

			//update (crop out) template images
			//referenced from: http://opencv.willowgarage.com/documentation/cpp/c++_cheatsheet.html
			
			//top
			//Rect topROI(topLoc);	//Make a rectangle
			//Mat imgTopROI = Mat(img)(topROI);	//Point a cv::Mat header at it (no allocation is done)
			//imgTopROI.copyTo(this->mouthTopTpl);

			cvNamedWindow( "mouthTop", 1 );
			imshow( "mouthTop", this->mouthTopTpl);


			//bottom
			//Rect bottomROI(bottomLoc);	//Make a rectangle
			//Mat imgBottomROI = Mat(img)(bottomROI);	//Point a cv::Mat header at it (no allocation is done)
			//imgBottomROI.copyTo(this->mouthBottomTpl);

			cvNamedWindow( "mouthBottom", 1 );
			imshow( "mouthBottom", this->mouthBottomTpl);

			//std::cout << "top bot diff: " << abs(topLoc.y - bottomLoc.y) << std::endl;

			return true;

		}
		else //clear out templates
		{
			this->mouthTopTpl.release();
			this->mouthBottomTpl.release();
			//we didnt' find the feature in this frame :(

			return true;
		}

		//std::cout << "top bot diff: " << abs(topLoc.y - bottomLoc.y) << std::endl;


	}//end updateMouthSubFeatureLocs


	/* Look in the lower half of the face for the mouth */
	CvRect getMouthSearchSpace(CvRect *r)
	{
		return cvRect(r->x,
						(r->y +  (int)((1.0/2.0)*r->height)),
						r->width, 
						(int)((1.0/2.0)*r->height));
	}

	/* look in top left 5/8's for left eyebrow */
	CvRect getLeftEyebrowSearchSpace(CvRect *r)
	{
		return cvRect(r->x,
						r->y, 
						(int)((5.0/8.0)*r->width), 
						(int)((1.0/2.0)*r->height) );
	}

	/* look in top right 5/8's for right eyebrow */
	CvRect getRightEyebrowSearchSpace(CvRect *r)
	{
		return cvRect(r->x + (int)((3.0/8.0)*r->width),
						r->y,
						(int)((5.0/8.0)*r->width),
						(int)((1.0/2.0)*r->height) );
	}

	/* look under left eyebrow midpoint and face midpoint for left eye */
	CvRect getLeftEyeSearchSpace(CvRect *r)
	{
		int y = this->leftEyebrowLoc.y + (int)((1.0/2.0)*this->leftEyebrowLoc.height);
		return cvRect(r->x,
						y, 
						(int)((1.0/2.0)*r->width), 
						this->mouthLoc.y - y );
	}

	/* look under right eyebrow midpoint and face midpoint for right eye */
	CvRect getRightEyeSearchSpace(CvRect *r)
	{
		int y = this->rightEyebrowLoc.y + (int)((1.0/2.0)*this->rightEyebrowLoc.height);
		return cvRect(r->x + (int)((1.0/2.0)*r->width),
						y,
						(int)((1.0/2.0)*r->width),
						this->mouthLoc.y - y );
	}


	/* Crop out a template from a face */
	void cropTemplate(IplImage *img, CvRect& loc, Mat& dst)
	{
		Rect ROI(loc);	//Make a rectangle
		Mat imgROI = Mat(img)(ROI);	//Point a cv::Mat header at it (no allocation is done)
		imgROI.copyTo(dst);
	}


	/* Update locations of all subfeatures */
	bool updateSubFeatureLocs(IplImage *img, IplImage *processedImg, CvRect *r)
	{
		//update face coordinates
		this->updateFaceCoords(r);

		//mouth
		//bool updateMouthSuccess = this->updateMouthSubFeatureLocs(img,processedImg,r);
		bool updateMouthSuccess = true;
		return (updateMouthSuccess);
	}

	/* Update locations of all features by running NCC's */
	bool updateFeatureLocs(IplImage *img, IplImage *processedImg, CvRect *r)
	{
		//call isValidFace without cropping out templates
		return this->isValidFace(img,processedImg,r,false);
	}


	/* Return true if a face found by Haar is valid (has mouth,eyebrows,eyes) */
	bool isValidFace(IplImage *img, IplImage *processedImg, CvRect *r, bool doCrop = true)
	{
		//update face coordinates
		this->updateFaceCoords(r);

		//store face dimensions in order to resize templates
		double newFaceWidth = r->width;
		double newFaceHeight = r->height;
		
		//return false if any of the features are not found

		/* MOUTH */
		Mat mouthTpl;
		if(this->mouthTpl.empty())	//use default template from average face
		{
			Mat oldMouthTpl	= imread("templates/mouth.jpg",1);
			resizeFeatureTemplate(oldMouthTpl,newFaceWidth,newFaceHeight,mouthTpl);
		}
		else //use an existing template from this face 
			//(this only happens in debugging when we run isValidFace even on old faces)
		{
			mouthTpl = this->mouthTpl;
		}
		
		//run NCC to get coordinates of the mouth
		CvRect mouthSearchSpace = this->getMouthSearchSpace(r);
		CvRect mouthLoc;
		//std::cout << "mouth=============================" << std::endl;
		bool mouthFound = getSearchSpace(img,processedImg,r,mouthTpl,mouthSearchSpace,mouthLoc,true);
		//std::cout << "==============================mouth" << std::endl;
		
		//break out if no mouth found
		if(mouthFound == false)
		{
			return false;
		}
		//store mouth coordinates and template image 
		//(at first this will just be the resized version of the average face template)
		else
		{
			this->mouthTpl = mouthTpl;
			this->mouthLoc = mouthLoc;	
		}
		/* END MOUTH */

		/* LEFT EYEBROW */
		Mat leftEyebrowTpl;
		if(this->leftEyebrowTpl.empty())
		{
			Mat oldLeftEyebrowTpl = imread("templates/leftEyebrow.jpg",1);
			resizeFeatureTemplate(oldLeftEyebrowTpl,newFaceWidth,newFaceHeight,leftEyebrowTpl);
		}
		else
		{
			leftEyebrowTpl = this->leftEyebrowTpl;
		}
		
		CvRect leftEyebrowSearchSpace = this->getLeftEyebrowSearchSpace(r);
		CvRect leftEyebrowLoc;
		//std::cout << "left eyebrow =============================" << std::endl;
		bool leftEyebrowFound = getSearchSpace(img,processedImg,r,leftEyebrowTpl,leftEyebrowSearchSpace,leftEyebrowLoc,true);
		//std::cout << " ============================= left eyebrow" << std::endl;

		if(leftEyebrowFound == false)
		{
			return false;
		}
		else //update left eyebrow template
		{
			this->leftEyebrowTpl = leftEyebrowTpl;
			this->leftEyebrowLoc = leftEyebrowLoc;
		}
		/* END LEFT EYEBROW */

		/* RIGHT EYEBROW */
		Mat rightEyebrowTpl;

		if(this->rightEyebrowTpl.empty())
		{
			Mat oldRightEyebrowTpl = imread("templates/rightEyebrow.jpg",1);
			resizeFeatureTemplate(oldRightEyebrowTpl,newFaceWidth,newFaceHeight,rightEyebrowTpl);
		}
		else
		{
			rightEyebrowTpl = this->rightEyebrowTpl;
		}
		
		CvRect rightEyebrowSearchSpace = this->getRightEyebrowSearchSpace(r);
		CvRect rightEyebrowLoc;
		//std::cout << "right eyebrow =============================" << std::endl;
		bool rightEyebrowFound = getSearchSpace(img,processedImg,r,rightEyebrowTpl,rightEyebrowSearchSpace,rightEyebrowLoc,true);
		//std::cout << " ============================= right eyebrow" << std::endl;

		if(rightEyebrowFound == false)
		{
			return false;
		}
		else //update right eyebrow template
		{
			this->rightEyebrowTpl = rightEyebrowTpl;
			this->rightEyebrowLoc = rightEyebrowLoc;
		}
		/* END RIGHT EYEBROW */

		/* LEFT EYE */
		Mat leftEyeTpl;
		if(this->leftEyeTpl.empty())
		{
			Mat oldLeftEyeTpl = imread("templates/leftEye.jpg",1);
			resizeFeatureTemplate(oldLeftEyeTpl,newFaceWidth,newFaceHeight,leftEyeTpl);
		}
		else
		{
			leftEyeTpl = this->leftEyeTpl;
		}
		
		CvRect leftEyeSearchSpace = this->getLeftEyeSearchSpace(r);
		CvRect leftEyeLoc;
		//std::cout << "left eye =============================" << std::endl;
		bool leftEyeFound = getSearchSpace(img,processedImg,r,leftEyeTpl,leftEyeSearchSpace,leftEyeLoc,true);
		//std::cout << "=============================left eye" << std::endl;
		
		if(leftEyeFound == false)
		{
			return false;
		}
		else //update left eye stuff
		{
			this->leftEyeTpl = leftEyeTpl;
			this->leftEyeLoc = leftEyeLoc;
		}
		/* END LEFT EYE */

		/* RIGHT EYE */
		Mat rightEyeTpl;
		if(this->rightEyeTpl.empty())
		{
			Mat oldRightEyeTpl = imread("templates/rightEye.jpg",1);
			resizeFeatureTemplate(oldRightEyeTpl,newFaceWidth,newFaceHeight,rightEyeTpl);
		}
		else
		{
			rightEyeTpl = this->rightEyeTpl;
		}

		CvRect rightEyeSearchSpace = this->getRightEyeSearchSpace(r);
		CvRect rightEyeLoc;
		//std::cout << "right eye=============================" << std::endl;
		bool rightEyeFound = getSearchSpace(img,processedImg,r,rightEyeTpl,rightEyeSearchSpace,rightEyeLoc,true);
		//std::cout << "=============================right eye" << std::endl;

		if(rightEyeFound == false)
		{
			return false;
		}
		else //update right eye stuff
		{
			this->rightEyeTpl = rightEyeTpl;
			this->rightEyeLoc = rightEyeLoc;
		}
		/* END RIGHT EYE */

		//crop out all feature templates from the valid face
		//we do all of these at the end so we don't waste time cropping out features from invalid faces
		if(doCrop == true)
		{
			this->cropTemplate(img,this->mouthLoc,this->mouthTpl);
			this->cropTemplate(img,this->leftEyebrowLoc,this->leftEyebrowTpl);
			this->cropTemplate(img,this->rightEyebrowLoc,this->rightEyebrowTpl);
			this->cropTemplate(img,this->leftEyeLoc,this->leftEyeTpl);
			this->cropTemplate(img,this->rightEyeLoc,this->rightEyeTpl);
		}

		//update sub templates			
		//updateMouthSubFeatureLocs(img, processedImg, *r, mouthLoc);
		//updateMouthSubFeatureLocs(img, processedImg, *r);
		
		//updateSubFeatureLocs(img, processedImg, *r);

		return true;
	}

	void eyeSubCode()
	{
	//	/* Update locations of subfeatures by running NCC on them*/
	////void updateEyeSubFeatureLocations(string featureFilename, newFaceCoords, boolean doResize, boolean doCrop, int frame #)
	////void updateEyeSubFeatureLocations(IplImage *img, IplImage *processedImg, double newFaceWidth, double newFaceHeight, CvRect& parentLoc)
	//void updateEyeSubFeatureLocations(IplImage *img, IplImage *processedImg, CvRect& r, CvRect& parentLoc)
	//{
	//	//update face with new coordinates
	//	this->oldFaceWidth = this->currentFaceWidth;
	//	this->oldFaceHeight = this->currentFaceWidth;

	//	this->currentFaceWidth = r.width;	//newFaceWidth
	//	this->currentFaceHeight = r.height;	//newFaceHeight
	//	
	//	this->topLeftPoint.x = r.x;
	//	this->topLeftPoint.y = r.y;


	//	CvRect topLoc = cvRect(0,0,0,0);			
	//	CvRect bottomLoc = cvRect(0,0,0,0);

	//	CvRect topSearchSpace;		// if condition is true, parent feature subsection, else subfeature location with Buffer
	//	CvRect bottomSearchSpace;

	//	//subfeature has no previous coordinates, i.e. the template images are empty
	//	if(leftEyeTopTpl.empty() || leftEyeBottomTpl.empty())
	//	//if(true)
	//	{
	//		//look at parent feature coords for search space
	//		
	//		//crop out eye template
	//		Rect parentROI(parentLoc);	//Make a rectangle
	//		Mat imgParentROI = Mat(img)(parentROI);	//Point a cv::Mat header at it (no allocation is done)
	//		//imgParentROI.copyTo(this->leftEyeParentTpl);

	//		cvNamedWindow( "eyeTop", 1 );
	//		imshow( "eyeTop", imgParentROI);


	//		//top subtemplate
	//		//Mat oldTopTpl = imread("templates/leftEyeTop.jpg",1);			
	//		//resizeFeatureTemplate(oldTopTpl,r.width,r.height,this->leftEyeTopTpl);
	//		
	//		//topSearchSpace = cvRect(parentLoc.x, parentLoc.y,parentLoc.width, parentLoc.height/2);
	//		topSearchSpace = cvRect(parentLoc.x, parentLoc.y,parentLoc.width, (3.0/8.0)*parentLoc.height);


	//		Rect topROI(topSearchSpace);
	//		Mat imgTopROI = Mat(img)(topROI);
	//		imgTopROI.copyTo(this->leftEyeTopTpl);

	//		cvNamedWindow( "leftEyeTop", 1 );
	//		imshow( "leftEyeTop", this->leftEyeTopTpl);

	//		
	//		//yellow box
	//		//rectangle(Mat(processedImg),Point(topSearchSpace.x,topSearchSpace.y),
	//		//	Point(topSearchSpace.x+topSearchSpace.width,topSearchSpace.y+topSearchSpace.height),
	//		//	CV_RGB(255,255,0),1,0,0);


	//		//bottom subtemplate
	//		//Mat oldBottomTpl = imread("templates/leftEyeBottom.jpg",1);
	//		//resizeFeatureTemplate(oldBottomTpl,r.width,r.height,this->leftEyeBottomTpl);

	//		//bottomSearchSpace = cvRect(parentLoc.x, parentLoc.y + (1.0/2.0)*parentLoc.height, parentLoc.width, (1.0/2.0)*parentLoc.height);
	//		bottomSearchSpace = cvRect(parentLoc.x, parentLoc.y + (5.0/8.0)*parentLoc.height, parentLoc.width, (3.0/8.0)*parentLoc.height);

	//		Rect bottomROI(bottomSearchSpace);	//Make a rectangle
	//		Mat imgBottomROI = Mat(img)(bottomROI);	//Point a cv::Mat header at it (no allocation is done)
	//		imgBottomROI.copyTo(this->leftEyeBottomTpl);

	//		cvNamedWindow( "leftEyeBottom", 1 );
	//		imshow( "leftEyeBottom", this->leftEyeBottomTpl);


	//		//pink box
	//		//rectangle(Mat(processedImg),Point(bottomSearchSpace.x,bottomSearchSpace.y),
	//		//	Point(bottomSearchSpace.x+bottomSearchSpace.width,bottomSearchSpace.y+bottomSearchSpace.height),
	//		//	CV_RGB(255,0,255),1,0,0);

	//	}
	//	else //update search space for NCC
	//	{
 //////resize sub feature templates and search space based on new face width and height
 ////                       //double oldFaceWidth = (double)r.width;
 ////                       //double oldFaceHeight = (double)r.height;

 ////                       //top
 ////                       double oldTopTplWidth = (double)leftEyeTopTpl.cols;
 ////                       //double newTopTplWidth = (oldTopTplWidth / oldFaceWidth) * newFaceWidth;
 ////                       //double newTopTplWidth = (oldTopTplWidth / this->oldFaceWidth) * newFaceWidth;
 ////                       
 ////                       //double newTopTplWidth = (oldTopTplWidth / this->oldFaceWidth) * r.width;
 ////                       double newTopTplWidth = ceil( (oldTopTplWidth / this->oldFaceWidth) * r.width );

 ////                       //prevent against width <= 0
 ////                       if(newTopTplWidth < 1)
 ////                       {
 ////                             newTopTplWidth = 1;
 ////                       }

 ////                       std::cout << "old face width: " << this->oldFaceWidth << std::endl;
 ////                       std::cout << "new face width: " << r.width << std::endl;
 ////                       
 ////                       std::cout << "oldTopTplWidth: " << oldTopTplWidth << std::endl;
 ////                       std::cout << "newTopTplWidth: " << newTopTplWidth << std::endl;


 ////                       double oldTopTplHeight = (double)leftEyeTopTpl.rows;
 ////                       //double newTopTplHeight = (oldTopTplHeight / oldFaceHeight) * newFaceHeight;
 ////                       //double newTopTplHeight = (oldTopTplHeight / this->oldFaceHeight) * newFaceHeight;
 ////                       
 ////                       //double newTopTplHeight = (oldTopTplHeight / this->oldFaceHeight) * r.height;
 ////                       double newTopTplHeight = ceil( (oldTopTplHeight / this->oldFaceHeight) * r.height );
 ////                       
 ////                       //std::cout << "newTopTplHeight: " << newTopTplHeight << std::endl;

 ////                       //prevent against height <= 0
 ////                       if(newTopTplHeight < 1)
 ////                       {
 ////                             newTopTplHeight = 1;
 ////                       }

 ////                       std::cout << "old face height: " << this->oldFaceHeight << std::endl;
 ////                       std::cout << "new face height: " << r.height << std::endl;
 ////                       
 ////                       std::cout << "oldTopTplHeight: " << oldTopTplHeight << std::endl;
 ////                       std::cout << "newTopTplHeight: " << newTopTplHeight << std::endl;

 ////                       Mat newLeftEyeTopTpl;
 ////                       resize(this->leftEyeTopTpl,newLeftEyeTopTpl,Size((int)newTopTplWidth,(int)newTopTplHeight));

 ////                       //swap templates back                 
 ////                       //newLeftEyeTopTpl.copyTo(this->leftEyeTopTpl);
 ////                       this->leftEyeTopTpl = newLeftEyeTopTpl;


 ////                       //bottom
 ////                       double oldBottomTplWidth = (double)leftEyeBottomTpl.cols;
 ////                       //double newBottomTplWidth = (oldBottomTplWidth / oldFaceWidth) * newFaceWidth;
 ////                       //double newBottomTplWidth = (oldBottomTplWidth / this->oldFaceWidth) * newFaceWidth;
 ////                       double newBottomTplWidth = (oldBottomTplWidth / this->oldFaceWidth) * r.width;

	////					//std::cout << "newBottomTplWidth: " << newBottomTplWidth << std::endl;

 ////                       //prevent against width <= 0
 ////                       if(newBottomTplWidth < 1)
 ////                       {
 ////                             newBottomTplWidth = 1;
 ////                       }

 ////                       std::cout << "oldBottomTplWidth: " << oldBottomTplWidth << std::endl;
 ////                       std::cout << "newBottomTplWidth: " << newBottomTplWidth << std::endl;


 ////                       double oldBottomTplHeight = (double)leftEyeBottomTpl.rows;
 ////                       //double newBottomTplHeight = (oldBottomTplHeight / oldFaceHeight) * newFaceHeight;
 ////                       //double newBottomTplHeight = (oldBottomTplHeight / this->oldFaceHeight) * newFaceHeight;
 ////                       double newBottomTplHeight = (oldBottomTplHeight / this->oldFaceHeight) * r.height;
 ////                       //std::cout << "newBottomTplHeight: " << newBottomTplHeight << std::endl;

 ////                       //prevent against height <= 0
 ////                       if(newBottomTplHeight < 1)
 ////                       {
 ////                             newBottomTplHeight = 1;
 ////                       }

 ////                       std::cout << "oldBottomTplHeight: " << oldBottomTplHeight << std::endl;
 ////                       std::cout << "newBottomTplHeight: " << newBottomTplHeight << std::endl;

 ////                       Mat newLeftEyeBottomTpl;
 ////                       resize(this->leftEyeBottomTpl,newLeftEyeBottomTpl,Size((int)newBottomTplWidth,(int)newBottomTplHeight));

 ////                       //swap templates back                 
 ////                       //newLeftEyeBottomTpl.copyTo(this->leftEyeBottomTpl);
 ////                       this->leftEyeBottomTpl = newLeftEyeBottomTpl;
 ////                       //end resize



	//		//resize buffers for search space depending on how much face size changed

	//		int bufferX = (int)( (((double)r.width/(double)this->oldFaceWidth) * this->leftEyeTopLoc.x) - this->leftEyeTopLoc.x );
	//		//when the face grows smaller, just use no buffer (same search space as previous) to search instead of shrinking the search space
	//		//b/c the current template may actually be larger than a resized smaller search space
	//		if(bufferX < 0)
	//		{
	//			bufferX = 0;
	//		}
	//		
	//		int bufferY = (int)( (((double)r.height/(double)this->oldFaceHeight) * this->leftEyeTopLoc.y) - this->leftEyeTopLoc.y );
	//		if(bufferY < 0)
	//		{
	//			bufferY = 0;
	//		}

	//		//edge cases for top search space
	//		//use same maximum boundaries as when searching for whole eye template in face
	//		int topSearchSpaceX = leftEyeTopLoc.x - bufferX;
	//		if(topSearchSpaceX < this->topLeftPoint.x)
	//		{
	//			topSearchSpaceX = this->topLeftPoint.x;
	//		}

	//		int topSearchSpaceY = leftEyeTopLoc.y - bufferY;
	//		if(topSearchSpaceY < (this->topLeftPoint.y + (int)((2.5/8.0)*this->currentFaceHeight)) )
	//		{
	//			topSearchSpaceY = (this->topLeftPoint.y + (int)((2.5/8.0)*this->currentFaceHeight));
	//		}

	//		int topSearchSpaceWidth = this->leftEyeTopTpl.cols + 2*bufferX;
	//		if(topSearchSpaceWidth > (this->currentFaceWidth/2))
	//		{
	//			topSearchSpaceWidth = (this->currentFaceWidth/2);
	//		}

	//		int topSearchSpaceHeight = this->leftEyeTopTpl.rows + 2*bufferY;
	//		if(topSearchSpaceHeight > ((int)((2.0/8.0)*this->currentFaceHeight)) )
	//		{
	//			topSearchSpaceHeight = ((int)((2.0/8.0)*this->currentFaceHeight));
	//		}

	//		topSearchSpace = cvRect(topSearchSpaceX, topSearchSpaceY, topSearchSpaceWidth, topSearchSpaceHeight);

	//		//black rectangle for top search space
	//		//rectangle(Mat(processedImg),Point(topSearchSpace.x,topSearchSpace.y),
	//		//	Point(topSearchSpace.x + topSearchSpace.width, topSearchSpace.y + topSearchSpace.height),CV_RGB(0, 0, 0), 1, 0, 0 );


	//		//std::cout << "top search space: " << topSearchSpace.x << "," << topSearchSpace.y << ": " << topSearchSpace.width << "by" << topSearchSpace.height << std::endl;

	//		//edge cases for bottom search space
	//		//use same maximum boundaries as when searching for whole eye template in face
	//		int bottomSearchSpaceX = leftEyeBottomLoc.x - bufferX;
	//		if(bottomSearchSpaceX < this->topLeftPoint.x)
	//		{
	//			bottomSearchSpaceX = this->topLeftPoint.x;
	//		}

	//		int bottomSearchSpaceY = leftEyeBottomLoc.y - bufferY;
	//		if(bottomSearchSpaceY < (this->topLeftPoint.y + (int)((2.5/8.0)*this->currentFaceHeight)) )
	//		{
	//			bottomSearchSpaceY = (this->topLeftPoint.y + (int)((2.5/8.0)*this->currentFaceHeight));
	//		}

	//		int bottomSearchSpaceWidth = this->leftEyeBottomTpl.cols + 2*bufferX;
	//		if(bottomSearchSpaceWidth > (this->currentFaceWidth/2))
	//		{
	//			bottomSearchSpaceWidth = (this->currentFaceWidth/2);
	//		}

	//		int bottomSearchSpaceHeight = this->leftEyeBottomTpl.rows + 2*bufferY;
	//		if(bottomSearchSpaceHeight > ((int)((2.0/8.0)*this->currentFaceHeight)) )
	//		{
	//			bottomSearchSpaceHeight = ((int)((2.0/8.0)*this->currentFaceHeight));
	//		}

	//		bottomSearchSpace = cvRect(bottomSearchSpaceX, bottomSearchSpaceY, bottomSearchSpaceWidth, bottomSearchSpaceHeight);

	//		//white box
	//	/*	rectangle(Mat(processedImg),Point(bottomSearchSpace.x,bottomSearchSpace.y),
	//			Point(bottomSearchSpace.x + bottomSearchSpace.width, bottomSearchSpace.y + bottomSearchSpace.height),CV_RGB(255, 255, 255), 1, 0, 0 );*/

	//	}//end else


	//	//update template image by running NCC
	//	bool topFound = getSearchSpace(img,processedImg,&r,leftEyeTopTpl,topSearchSpace,topLoc);
	//	bool bottomFound = getSearchSpace(img,processedImg,&r,leftEyeBottomTpl,bottomSearchSpace,bottomLoc);

	//	//update coordinates because maybe it was only taking the greatest but not surpassing threshold
	//	//if(topFound && bottomFound)
	//	if(true)
	//	{
	//		this->leftEyeTopLoc.x = topLoc.x;
	//		this->leftEyeTopLoc.y = topLoc.y;
	//		this->leftEyeBottomLoc.x = bottomLoc.x;
	//		this->leftEyeBottomLoc.y = bottomLoc.y;

	//		//update (crop out) template images
	//		//referenced from: http://opencv.willowgarage.com/documentation/cpp/c++_cheatsheet.html
	//		
	//		//top
	//		//Rect topROI(topLoc);	//Make a rectangle
	//		//Mat imgTopROI = Mat(img)(topROI);	//Point a cv::Mat header at it (no allocation is done)
	//		//imgTopROI.copyTo(this->leftEyeTopTpl);

	//		cvNamedWindow( "leftEyeTop", 1 );
	//		imshow( "leftEyeTop", this->leftEyeTopTpl);


	//		//bottom
	//		//Rect bottomROI(bottomLoc);	//Make a rectangle
	//		//Mat imgBottomROI = Mat(img)(bottomROI);	//Point a cv::Mat header at it (no allocation is done)
	//		//imgBottomROI.copyTo(this->leftEyeBottomTpl);

	//		cvNamedWindow( "leftEyeBottom", 1 );
	//		imshow( "leftEyeBottom", this->leftEyeBottomTpl);

	//		//std::cout << "top bot diff: " << abs(topLoc.y - bottomLoc.y) << std::endl;

	//	}
	//	else //clear out templates
	//	{
	//		this->leftEyeTopTpl.release();
	//		this->leftEyeBottomTpl.release();
	//		//we didnt' find the feature in this frame :(
	//	}

	//	std::cout << "top bot diff: " << abs(topLoc.y - bottomLoc.y) << std::endl;


	//}//end updateEyeSubFeatureLocations


	//
	//

	}


	void resizeCode()
	{
		 ////resize sub feature templates and search space based on new face width and height
   //                     //double oldFaceWidth = (double)r.width;
   //                     //double oldFaceHeight = (double)r.height;

   //                     //top
   //                     double oldTopTplWidth = (double)mouthTopTpl.cols;
   //                     //double newTopTplWidth = (oldTopTplWidth / oldFaceWidth) * newFaceWidth;
   //                     //double newTopTplWidth = (oldTopTplWidth / this->oldFaceWidth) * newFaceWidth;
   //                     
   //                     //double newTopTplWidth = (oldTopTplWidth / this->oldFaceWidth) * r.width;
   //                     double newTopTplWidth = ceil( (oldTopTplWidth / this->oldFaceWidth) * r.width );

   //                     //prevent against width <= 0
   //                     if(newTopTplWidth < 1)
   //                     {
   //                           newTopTplWidth = 1;
   //                     }

   //                     std::cout << "old face width: " << this->oldFaceWidth << std::endl;
   //                     std::cout << "new face width: " << r.width << std::endl;
   //                     
   //                     std::cout << "oldTopTplWidth: " << oldTopTplWidth << std::endl;
   //                     std::cout << "newTopTplWidth: " << newTopTplWidth << std::endl;


   //                     double oldTopTplHeight = (double)mouthTopTpl.rows;
   //                     //double newTopTplHeight = (oldTopTplHeight / oldFaceHeight) * newFaceHeight;
   //                     //double newTopTplHeight = (oldTopTplHeight / this->oldFaceHeight) * newFaceHeight;
   //                     
   //                     //double newTopTplHeight = (oldTopTplHeight / this->oldFaceHeight) * r.height;
   //                     double newTopTplHeight = ceil( (oldTopTplHeight / this->oldFaceHeight) * r.height );
   //                     
   //                     //std::cout << "newTopTplHeight: " << newTopTplHeight << std::endl;

   //                     //prevent against height <= 0
   //                     if(newTopTplHeight < 1)
   //                     {
   //                           newTopTplHeight = 1;
   //                     }

   //                     std::cout << "old face height: " << this->oldFaceHeight << std::endl;
   //                     std::cout << "new face height: " << r.height << std::endl;
   //                     
   //                     std::cout << "oldTopTplHeight: " << oldTopTplHeight << std::endl;
   //                     std::cout << "newTopTplHeight: " << newTopTplHeight << std::endl;

   //                     Mat newMouthTopTpl;
   //                     resize(this->mouthTopTpl,newMouthTopTpl,Size((int)newTopTplWidth,(int)newTopTplHeight));

   //                     //swap templates back                 
   //                     //newMouthTopTpl.copyTo(this->mouthTopTpl);
   //                     this->mouthTopTpl = newMouthTopTpl;


   //                     //bottom
   //                     double oldBottomTplWidth = (double)mouthBottomTpl.cols;
   //                     //double newBottomTplWidth = (oldBottomTplWidth / oldFaceWidth) * newFaceWidth;
   //                     //double newBottomTplWidth = (oldBottomTplWidth / this->oldFaceWidth) * newFaceWidth;
   //                     double newBottomTplWidth = (oldBottomTplWidth / this->oldFaceWidth) * r.width;

			//			//std::cout << "newBottomTplWidth: " << newBottomTplWidth << std::endl;

   //                     //prevent against width <= 0
   //                     if(newBottomTplWidth < 1)
   //                     {
   //                           newBottomTplWidth = 1;
   //                     }

   //                     std::cout << "oldBottomTplWidth: " << oldBottomTplWidth << std::endl;
   //                     std::cout << "newBottomTplWidth: " << newBottomTplWidth << std::endl;


   //                     double oldBottomTplHeight = (double)mouthBottomTpl.rows;
   //                     //double newBottomTplHeight = (oldBottomTplHeight / oldFaceHeight) * newFaceHeight;
   //                     //double newBottomTplHeight = (oldBottomTplHeight / this->oldFaceHeight) * newFaceHeight;
   //                     double newBottomTplHeight = (oldBottomTplHeight / this->oldFaceHeight) * r.height;
   //                     //std::cout << "newBottomTplHeight: " << newBottomTplHeight << std::endl;

   //                     //prevent against height <= 0
   //                     if(newBottomTplHeight < 1)
   //                     {
   //                           newBottomTplHeight = 1;
   //                     }

   //                     std::cout << "oldBottomTplHeight: " << oldBottomTplHeight << std::endl;
   //                     std::cout << "newBottomTplHeight: " << newBottomTplHeight << std::endl;

   //                     Mat newMouthBottomTpl;
   //                     resize(this->mouthBottomTpl,newMouthBottomTpl,Size((int)newBottomTplWidth,(int)newBottomTplHeight));

   //                     //swap templates back                 
   //                     //newMouthBottomTpl.copyTo(this->mouthBottomTpl);
   //                     this->mouthBottomTpl = newMouthBottomTpl;
   //                     //end resize
	}
	
};

#endif
