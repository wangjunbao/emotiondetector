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
	double oldFaceWidth;
	double oldFaceHeight;
	double currentFaceWidth;
	double currentFaceHeight;



	//mouth
	Mat mouthTpl;
	CvRect mouthLoc;

	Mat mouthTopTpl;
	CvRect mouthTopLoc;

	Mat mouthBottomTpl;
	CvRect mouthBottomLoc;

	Mat mouthLeftTpl;
	CvRect mouthLeftLoc;

	Mat mouthRightTpl;
	CvRect mouthRightLoc;


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

	//neutral feature positions                            
	double nBrowDistance; 
	double nLeftEyebrowRaised;
	double nRightEyebrowRaised;
	double nMouthOpen;
	double nMouthSmile;
	double nMouthFrown;

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

	double getWidth()
	{
		return currentFaceWidth;
	}

	double getHeight()
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

		////left eyebrow
		//namedWindow( "left eyebrow", 1 );
		//imshow("left eyebrow",this->leftEyebrowTpl);

		////right eyebrow
		//namedWindow( "right eyebrow", 1 );
		//imshow("right eyebrow",this->rightEyebrowTpl);

		////left eye
		//namedWindow( "left eye", 1 );
		//imshow("left eye",this->leftEyeTpl);

		////right eye
		//namedWindow( "right eye", 1 );
		//imshow("right eye",this->rightEyeTpl);

		////mouth
		//namedWindow( "mouth", 1 );
		//imshow("mouth",this->mouthTpl);

		////mouth top
		//namedWindow( "mouthTop", 1 );
		//imshow( "mouthTop", this->mouthTopTpl);

		////mouth bottom
		//cvNamedWindow( "mouthBottom", 1 );
		//imshow( "mouthBottom", this->mouthBottomTpl);

		////mouth left
		//cvNamedWindow( "mouthLeft", 1 );
		//imshow( "mouthLeft", this->mouthLeftTpl);

		////mouth right
		//cvNamedWindow( "mouthRight", 1 );
		//imshow( "mouthRight", this->mouthRightTpl);

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
		//std::cout << "===================" << std::endl;
		//std::cout << "Mat(img) size before: " << Mat(img).rows << " by " << Mat(img).cols << std::endl;
		//std::cout << "tpl size before: " << tpl.rows << " by " << tpl.cols << std::endl;
		//std::cout << "res size before: " << res.rows << " by " << res.cols << std::endl;
		//tpl.copyTo(res);//added to fix bug?

		try
		{
			matchTemplate(Mat(img), tpl, res, CV_TM_CCOEFF_NORMED);

			//std::cout << "Mat(img) size after: " << Mat(img).rows << " by " << Mat(img).cols << std::endl;
			//std::cout << "tpl size after: " << tpl.rows << " by " << tpl.cols << std::endl;
			//std::cout << "res size after: " << res.rows << " by " << res.cols << std::endl;
			//std::cout << "===================" << std::endl;
			
			////matimg
			//namedWindow( "matimg", 1 );
			//imshow("matimg",Mat(img));

			////tpl
			//namedWindow( "tpl", 1 );
			//imshow("tpl",tpl);

			////res
			//namedWindow( "res", 1 );
			//imshow("res",res);


			/* find best matches location */
			Point minloc, maxloc;
			double minval = 0.0;
			double maxval = 0.0;

			minMaxLoc(res, &minval, &maxval, &minloc, &maxloc); 

			//draw a box around the found feature even if its below the thresh
			//yellow box
			if(detailedOutput == true)
			{
				rectangle(Mat(processedImg),maxloc,Point(maxloc.x + tpl.cols, maxloc.y + tpl.rows),CV_RGB(255, 255, 0), 1, 0, 0 );
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

			//draw a box around the found feature
			//green box
			if(result == true && detailedOutput == true)
			{
				rectangle(Mat(processedImg),maxloc,Point(maxloc.x + tpl.cols, maxloc.y + tpl.rows),CV_RGB(0, 255, 0), 1, 0, 0 );
			}

			return result;
		}
		catch(int i)
		{
			std::cout << "EXCEPTION IN GET SEARCH SPACE: int: " << i << std::endl;
			return false;
		}
		catch(char *str)
		{
			std::cout << "EXCEPTION IN GET SEARCH SPACE: str: " << str << std::endl;
			return false;
		}
		catch(...)
		{
			std::cout << "EXCEPTION IN GET SEARCH SPACE: UNKNOWN" << std::endl;
			return false;

		}


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





	/* Look in the lower half of the face for the mouth */
	CvRect getMouthSearchSpace(CvRect *r)
	{
		return cvRect(r->x,
						(r->y +  (int)((2.0/3.0)*r->height)),
						r->width, 
						(int)((1.0/3.0)*r->height));
	}

	/* Look in the top halfish area of the mouth for the mouthTop */
	CvRect getMouthTopSearchSpace()
	{
		CvRect parentLoc = this->mouthLoc;
		return cvRect(parentLoc.x + (int)((1.0/4.0)*parentLoc.width),
						parentLoc.y,
						(int)((1.0/2.0)*parentLoc.width),
						(int)((3.0/8.0)*parentLoc.height));	
	}

	/* Look in the bottom halfish area of the mouth for the mouthBottom */
	CvRect getMouthBottomSearchSpace()
	{
		CvRect parentLoc = this->mouthLoc;
		return cvRect(parentLoc.x + (int)((1.0/4.0)*parentLoc.width),
						parentLoc.y + (int)((5.0/8.0)*parentLoc.height), 
						(int)((1.0/2.0)*parentLoc.width),
						(int)((3.0/8.0)*parentLoc.height));	
	}

	/* Look in the left halfish area of the mouth for the mouthLeft */
	CvRect getMouthLeftSearchSpace()
	{
		CvRect parentLoc = this->mouthLoc;
		return cvRect(parentLoc.x,
						parentLoc.y + (int)((1.0/4.0)*parentLoc.height), 
						(int)((1.0/4.0)*parentLoc.width),
						(int)((1.0/2.0)*parentLoc.height));	
	}


	/* Look in the right halfish area of the mouth for the mouthRight */
	CvRect getMouthRightSearchSpace()
	{
		CvRect parentLoc = this->mouthLoc;
		return cvRect(parentLoc.x + (int)((3.0/4.0)*parentLoc.width),
						parentLoc.y + (int)((1.0/4.0)*parentLoc.height), 
						(int)((1.0/4.0)*parentLoc.width),
						(int)((1.0/2.0)*parentLoc.height));	
	}


	/* look in top left 5/8's for left eyebrow */
	CvRect getLeftEyebrowSearchSpace(CvRect *r)
	{
		return cvRect(r->x,
						r->y, 
						(int)((5.0/8.0)*r->width), 
						(int)((1.0/2.0)*r->height) );
	}

	/* look near previous location of left eyebrow */
	CvRect getUpdatedSearchSpace(CvRect *r, 
		CvRect &defaultSearchSpace, Mat& featureTpl, CvRect& featureLoc)
	{
		//CHANGE THESE FOR EACH FEATURE
		//CvRect defaultSearchSpace = this->getLeftEyebrowSearchSpace(r);
		//Mat featureTpl = this->leftEyebrowTpl;
		//CvRect featureLoc = this->leftEyebrowLoc;

		int bufferX = (int)((0.5)*featureTpl.rows);
		if(bufferX < 0)
		{
			bufferX = 0;
		}

		int bufferY = bufferX;
		if(bufferY < 0)
		{
			bufferY = 0;
		}

		//Edge cases for search spaces
		//use same maximum boundaries as default
	
		/* Edge Cases */
		int searchSpaceX = featureLoc.x - bufferX;
		if(searchSpaceX < defaultSearchSpace.x)
		{
			searchSpaceX = defaultSearchSpace.x;
		}

		int searchSpaceY = featureLoc.y - bufferY;
		if(searchSpaceY < defaultSearchSpace.y)
		{
			searchSpaceY = defaultSearchSpace.y;
		}

		//make sure search space is not wider than the face
		int searchSpaceWidth = featureTpl.cols + 2*bufferX;
		int rightEdge = searchSpaceX + searchSpaceWidth;
		int faceRightEdge = (int)(this->topLeftPoint.x + this->currentFaceWidth);
		
		if(rightEdge > faceRightEdge)
		{
			searchSpaceWidth = faceRightEdge - searchSpaceX;
		}

		int searchSpaceHeight = featureTpl.rows + 2*bufferY;
		int bottomEdge = searchSpaceY + searchSpaceHeight;
		int faceBottomEdge = (int)(this->topLeftPoint.y + this->currentFaceHeight);

		if(bottomEdge > faceBottomEdge)
		{
			searchSpaceHeight = faceBottomEdge - searchSpaceY;
		}
		/* End Edge Cases */
	
		return cvRect(searchSpaceX, searchSpaceY, searchSpaceWidth, searchSpaceHeight);
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


	/* Update the location of the mouth */
	bool updateMouth(IplImage *img, IplImage *processedImg, CvRect *r, bool detailedOutput)
	{
		//store face dimensions in order to resize templates
		double newFaceWidth = r->width;
		double newFaceHeight = r->height;

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
		bool mouthFound = getSearchSpace(img,processedImg,r,mouthTpl,mouthSearchSpace,mouthLoc,detailedOutput);
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
		return true; //update was successful
	}


	/* 
		Update locations of mouth subfeatures
		Return true if update successful, false if unsuccessful (ie: features were lost)
	*/
	bool updateMouthTopBottom(IplImage *img, IplImage *processedImg, CvRect *r)
	{
		CvRect topSearchSpace;
		CvRect bottomSearchSpace;
			
		//using twice the number of rows seems to give a good space
		int bufferX = (int)((0.5)*this->mouthTopTpl.rows);
		//std::cout << "buffer x: " << bufferX << std::endl;
		if(bufferX < 0)
		{
			bufferX = 0;
		}
		
		//int bufferY = this->mouthTopTpl.rows + 
		//	(int)( (((double)r->height/(double)this->oldFaceHeight) * this->mouthTopLoc.y) - this->mouthTopLoc.y );
		//int bufferY = 15;//10;//5;			
		int bufferY = bufferX; //2*this->mouthTopTpl.rows;//(int)(1.5*this->mouthTopTpl.rows);
		if(bufferY < 0)
		{
			bufferY = 0;
		}

		//Edge cases for search spaces
		//use same maximum boundaries as when searching for whole mouth template in face
		CvRect mouthSearchSpace = this->getMouthSearchSpace(r);
		//it is possible for the bottom lip to leave the "face" when the mouth is opened really wide
		//but we rarely run into this case because the search space will not exceed half the height of the face

		/* Top Edge Cases */
		int topSearchSpaceX = mouthTopLoc.x - bufferX;
		if(topSearchSpaceX < mouthSearchSpace.x)
		{
			topSearchSpaceX = mouthSearchSpace.x;
		}

		int topSearchSpaceY = mouthTopLoc.y - bufferY;
		if(topSearchSpaceY < mouthSearchSpace.y)
		{
			topSearchSpaceY = mouthSearchSpace.y;
		}

		//make sure search space is not wider than the face
		int topSearchSpaceWidth = this->mouthTopTpl.cols + 2*bufferX;
		
		int topRightEdge = topSearchSpaceX + topSearchSpaceWidth;
		int faceRightEdge = (int)(this->topLeftPoint.x + this->currentFaceWidth);
		
		if(topRightEdge > faceRightEdge)
		//if(topSearchSpaceWidth > mouthSearchSpace.width)
		{
			//topSearchSpaceWidth = mouthSearchSpace.width;
			topSearchSpaceWidth = faceRightEdge - topSearchSpaceX;
		}

		int topSearchSpaceHeight = this->mouthTopTpl.rows + 2*bufferY;

		int topBottomEdge = topSearchSpaceY + topSearchSpaceHeight;
		int faceBottomEdge = (int)(this->topLeftPoint.y + this->currentFaceHeight);

		if(topBottomEdge > faceBottomEdge)
		//if(topSearchSpaceHeight > mouthSearchSpace.height)
		{
			topSearchSpaceHeight = faceBottomEdge - topSearchSpaceY;
			//topSearchSpaceHeight = mouthSearchSpace.height;
		}
		/* End Top Edge Cases */

		topSearchSpace = cvRect(topSearchSpaceX, topSearchSpaceY, topSearchSpaceWidth, topSearchSpaceHeight);

		//black rectangle for top search space
		rectangle(Mat(processedImg),Point(topSearchSpace.x,topSearchSpace.y),
			Point(topSearchSpace.x + topSearchSpace.width, topSearchSpace.y + topSearchSpace.height),CV_RGB(0, 0, 0), 1, 0, 0 );


		/* Bottom Edge Cases */
		int bottomSearchSpaceX = mouthBottomLoc.x - bufferX;
		if(bottomSearchSpaceX < mouthSearchSpace.x)
		{
			bottomSearchSpaceX = mouthSearchSpace.x;
		}

		int bottomSearchSpaceY = mouthBottomLoc.y - bufferY;
		if(bottomSearchSpaceY < mouthSearchSpace.y)
		{
			bottomSearchSpaceY = mouthSearchSpace.y;
		}

		int bottomSearchSpaceWidth = this->mouthBottomTpl.cols + 2*bufferX;
		int bottomRightEdge = bottomSearchSpaceX + bottomSearchSpaceWidth;

		if(bottomRightEdge > faceRightEdge)
		//if(bottomSearchSpaceWidth > mouthSearchSpace.width)
		{
			bottomSearchSpaceWidth = faceRightEdge - bottomSearchSpaceX;
			//bottomSearchSpaceWidth = mouthSearchSpace.width;
		}

		int bottomSearchSpaceHeight = this->mouthBottomTpl.rows + 2*bufferY;
		int bottomBottomEdge = bottomSearchSpaceY + bottomSearchSpaceHeight;

		if(bottomBottomEdge > (int)((1.25)*faceBottomEdge))
		//if(bottomSearchSpaceHeight > mouthSearchSpace.height)
		{
			bottomSearchSpaceHeight = (int)((1.25)*faceBottomEdge) - bottomSearchSpaceY;
			//bottomSearchSpaceHeight = mouthSearchSpace.height;
		}
		/* End Bottom Edge Cases */

		bottomSearchSpace = cvRect(bottomSearchSpaceX, bottomSearchSpaceY, bottomSearchSpaceWidth, bottomSearchSpaceHeight);

		//white box
		rectangle(Mat(processedImg),Point(bottomSearchSpace.x,bottomSearchSpace.y),
			Point(bottomSearchSpace.x + bottomSearchSpace.width, bottomSearchSpace.y + bottomSearchSpace.height),CV_RGB(255, 255, 255), 1, 0, 0 );

		//Update sub feature locations by running NCC
		

		CvRect topLoc;
		CvRect bottomLoc;
		bool topFound;
		bool bottomFound;

		/* UPDATE USING BUFFERS */
		topFound = getSearchSpace(img,processedImg,r,mouthTopTpl,topSearchSpace,topLoc,true);
		bottomFound = getSearchSpace(img,processedImg,r,mouthBottomTpl,bottomSearchSpace,bottomLoc,true);
		if(topFound == true && bottomFound == true)
		//if(true)
		{
			this->mouthTopLoc = topLoc;
			this->mouthBottomLoc = bottomLoc;
			return true;
		}
		/* END UPDATE USING BUFFERS */


		/* LAST DITCH: UPDATE USING DEFAULT SEARCH SPACE: LOOK WHERE MOUTH IS */

		//update mouth first
		this->updateMouth(img,processedImg,r,false);

		std::cout << "last ditch" << std::endl;
		
		topSearchSpace = this->getMouthTopSearchSpace();
		bottomSearchSpace = this->getMouthBottomSearchSpace();
		
		topFound = getSearchSpace(img,processedImg,r,mouthTopTpl,topSearchSpace,topLoc,true);
		bottomFound = getSearchSpace(img,processedImg,r,mouthBottomTpl,bottomSearchSpace,bottomLoc,true);

		if(topFound == true && bottomFound == true)
		{
			this->mouthTopLoc = topLoc;
			this->mouthBottomLoc = bottomLoc;
			return true;
		}
		else //feature was lost, update failed
		{
			return false;
		}
	}//end updateMouthTopBottom


	bool updateMouthLeftRight(IplImage *img, IplImage *processedImg, CvRect *r)
	{
		CvRect leftSearchSpace;
		CvRect rightSearchSpace;
			
		//using twice the number of rows seems to give a good space
		int bufferX = (int)((0.5)*this->mouthLeftTpl.rows);
		//std::cout << "buffer x: " << bufferX << std::endl;
		if(bufferX < 0)
		{
			bufferX = 0;
		}
		
		//int bufferY = this->mouthLeftTpl.rows + 
		//	(int)( (((double)r->height/(double)this->oldFaceHeight) * this->mouthLeftLoc.y) - this->mouthLeftLoc.y );
		//int bufferY = 15;//10;//5;			
		int bufferY = bufferX; //2*this->mouthLeftTpl.rows;//(int)(1.5*this->mouthLeftTpl.rows);
		if(bufferY < 0)
		{
			bufferY = 0;
		}

		//Edge cases for search spaces
		//use same maximum boundaries as when searching for whole mouth template in face
		CvRect mouthSearchSpace = this->getMouthSearchSpace(r);
	
		/* Left Edge Cases */
		int leftSearchSpaceX = mouthLeftLoc.x - bufferX;
		if(leftSearchSpaceX < mouthSearchSpace.x)
		{
			leftSearchSpaceX = mouthSearchSpace.x;
		}

		int leftSearchSpaceY = mouthLeftLoc.y - bufferY;
		if(leftSearchSpaceY < mouthSearchSpace.y)
		{
			leftSearchSpaceY = mouthSearchSpace.y;
		}

		//make sure search space is not wider than the face
		int leftSearchSpaceWidth = this->mouthLeftTpl.cols + 2*bufferX;
		
		int leftRightEdge = leftSearchSpaceX + leftSearchSpaceWidth;
		int faceRightEdge = (int)(this->topLeftPoint.x + this->currentFaceWidth);
		
		if(leftRightEdge > faceRightEdge)
		//if(leftSearchSpaceWidth > mouthSearchSpace.width)
		{
			//leftSearchSpaceWidth = mouthSearchSpace.width;
			leftSearchSpaceWidth = faceRightEdge - leftSearchSpaceX;
		}

		int leftSearchSpaceHeight = this->mouthLeftTpl.rows + 2*bufferY;

		int leftBottomEdge = leftSearchSpaceY + leftSearchSpaceHeight;
		int faceBottomEdge = (int)(this->topLeftPoint.y + this->currentFaceHeight);

		if(leftBottomEdge > faceBottomEdge)
		//if(leftSearchSpaceHeight > mouthSearchSpace.height)
		{
			leftSearchSpaceHeight = faceBottomEdge - leftSearchSpaceY;
			//leftSearchSpaceHeight = mouthSearchSpace.height;
		}
		/* End Left Edge Cases */

		leftSearchSpace = cvRect(leftSearchSpaceX, leftSearchSpaceY, leftSearchSpaceWidth, leftSearchSpaceHeight);

		//black rectangle for left search space
		rectangle(Mat(processedImg),Point(leftSearchSpace.x,leftSearchSpace.y),
			Point(leftSearchSpace.x + leftSearchSpace.width, leftSearchSpace.y + leftSearchSpace.height),CV_RGB(0, 0, 0), 1, 0, 0 );


		/* Right Edge Cases */
		int rightSearchSpaceX = mouthRightLoc.x - bufferX;
		if(rightSearchSpaceX < mouthSearchSpace.x)
		{
			rightSearchSpaceX = mouthSearchSpace.x;
		}

		int rightSearchSpaceY = mouthRightLoc.y - bufferY;
		if(rightSearchSpaceY < mouthSearchSpace.y)
		{
			rightSearchSpaceY = mouthSearchSpace.y;
		}

		int rightSearchSpaceWidth = this->mouthRightTpl.cols + 2*bufferX;
		int rightRightEdge = rightSearchSpaceX + rightSearchSpaceWidth;

		if(rightRightEdge > faceRightEdge)
		//if(rightSearchSpaceWidth > mouthSearchSpace.width)
		{
			rightSearchSpaceWidth = faceRightEdge - rightSearchSpaceX;
			//rightSearchSpaceWidth = mouthSearchSpace.width;
		}

		int rightSearchSpaceHeight = this->mouthRightTpl.rows + 2*bufferY;
		int rightBottomEdge = rightSearchSpaceY + rightSearchSpaceHeight;

		if(rightBottomEdge > (int)((1.25)*faceBottomEdge))
		//if(rightSearchSpaceHeight > mouthSearchSpace.height)
		{
			rightSearchSpaceHeight = (int)((1.25)*faceBottomEdge) - rightSearchSpaceY;
			//rightSearchSpaceHeight = mouthSearchSpace.height;
		}
		/* End Right Edge Cases */

		rightSearchSpace = cvRect(rightSearchSpaceX, rightSearchSpaceY, rightSearchSpaceWidth, rightSearchSpaceHeight);

		//white box
		rectangle(Mat(processedImg),Point(rightSearchSpace.x,rightSearchSpace.y),
			Point(rightSearchSpace.x + rightSearchSpace.width, rightSearchSpace.y + rightSearchSpace.height),CV_RGB(255, 255, 255), 1, 0, 0 );

		//Update sub feature locations by running NCC
		

		CvRect leftLoc;
		CvRect rightLoc;
		bool leftFound;
		bool rightFound;

		/* UPDATE USING BUFFERS */
		leftFound = getSearchSpace(img,processedImg,r,mouthLeftTpl,leftSearchSpace,leftLoc,true);
		rightFound = getSearchSpace(img,processedImg,r,mouthRightTpl,rightSearchSpace,rightLoc,true);
		if(leftFound == true && rightFound == true)
		//if(true)
		{
			this->mouthLeftLoc = leftLoc;
			this->mouthRightLoc = rightLoc;
			return true;
		}
		/* END UPDATE USING BUFFERS */


		/* LAST DITCH: UPDATE USING DEFAULT SEARCH SPACE: LOOK WHERE MOUTH IS */

		//update mouth first
		this->updateMouth(img,processedImg,r,false);

		std::cout << "last ditch" << std::endl;
		
		leftSearchSpace = this->getMouthLeftSearchSpace();
		rightSearchSpace = this->getMouthRightSearchSpace();
		
		leftFound = getSearchSpace(img,processedImg,r,mouthLeftTpl,leftSearchSpace,leftLoc,true);
		rightFound = getSearchSpace(img,processedImg,r,mouthRightTpl,rightSearchSpace,rightLoc,true);

		if(leftFound == true && rightFound == true)
		{
			this->mouthLeftLoc = leftLoc;
			this->mouthRightLoc = rightLoc;
			return true;
		}
		else //feature was lost, update failed
		{
			return false;
		}
	}//end updateMouthLeftRight

	
	bool updateLeftEyebrow(IplImage *img, IplImage *processedImg, CvRect *r, bool useDefaultSearchSpace, bool detailedOutput)
	{
		//store face dimensions in order to resize templates
		double newFaceWidth = r->width;
		double newFaceHeight = r->height;

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
		
		CvRect leftEyebrowSearchSpace;
		if(useDefaultSearchSpace == true)
		{
			leftEyebrowSearchSpace = this->getLeftEyebrowSearchSpace(r);
		}
		else
		{
			leftEyebrowSearchSpace = this->getUpdatedSearchSpace(r,
				this->getLeftEyebrowSearchSpace(r),this->leftEyebrowTpl,this->leftEyebrowLoc);
		}
		
		CvRect leftEyebrowLoc;
		//std::cout << "left eyebrow =============================" << std::endl;
		bool leftEyebrowFound = getSearchSpace(img,processedImg,r,leftEyebrowTpl,leftEyebrowSearchSpace,leftEyebrowLoc,detailedOutput);
		//std::cout << " ============================= left eyebrow" << std::endl;

		if(leftEyebrowFound == false)
		{
			return false;
		}
		else //update left eyebrow template
		{
			this->leftEyebrowTpl = leftEyebrowTpl;
			this->leftEyebrowLoc = leftEyebrowLoc;
			return true;
		}
		/* END LEFT EYEBROW */

	}

	
	bool updateRightEyebrow(IplImage *img, IplImage *processedImg, CvRect *r, bool useDefaultSearchSpace, bool detailedOutput)
	{
		//store face dimensions in order to resize templates
		double newFaceWidth = r->width;
		double newFaceHeight = r->height;

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
		
		CvRect rightEyebrowSearchSpace;
		if(useDefaultSearchSpace == true)
		{
			rightEyebrowSearchSpace = this->getRightEyebrowSearchSpace(r);
		}
		else
		{
			rightEyebrowSearchSpace = this->getUpdatedSearchSpace(r,
				this->getRightEyebrowSearchSpace(r),this->rightEyebrowTpl,this->rightEyebrowLoc);
		}

		
		
		CvRect rightEyebrowLoc;
		//std::cout << "right eyebrow =============================" << std::endl;
		bool rightEyebrowFound = getSearchSpace(img,processedImg,r,rightEyebrowTpl,rightEyebrowSearchSpace,rightEyebrowLoc,detailedOutput);
		//std::cout << " ============================= right eyebrow" << std::endl;

		if(rightEyebrowFound == false)
		{
			return false;
		}
		else //update right eyebrow template
		{
			this->rightEyebrowTpl = rightEyebrowTpl;
			this->rightEyebrowLoc = rightEyebrowLoc;
			return true;
		}
		/* END RIGHT EYEBROW */
	}


	bool updateLeftEye(IplImage *img, IplImage *processedImg, CvRect *r, bool useDefaultSearchSpace, bool detailedOutput)
	{
		//store face dimensions in order to resize templates
		double newFaceWidth = r->width;
		double newFaceHeight = r->height;

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
		
		CvRect leftEyeSearchSpace;
		if(useDefaultSearchSpace == true)
		{
			leftEyeSearchSpace = this->getLeftEyeSearchSpace(r);
		}
		else
		{
			leftEyeSearchSpace = this->getUpdatedSearchSpace(r,
				this->getLeftEyeSearchSpace(r),this->leftEyeTpl,this->leftEyeLoc);
		}


		CvRect leftEyeLoc;
		//std::cout << "left eye =============================" << std::endl;
		bool leftEyeFound = getSearchSpace(img,processedImg,r,leftEyeTpl,leftEyeSearchSpace,leftEyeLoc,detailedOutput);
		//std::cout << "=============================left eye" << std::endl;
		
		if(leftEyeFound == false)
		{
			return false;
		}
		else //update left eye stuff
		{
			this->leftEyeTpl = leftEyeTpl;
			this->leftEyeLoc = leftEyeLoc;
			return true;
		}
		/* END LEFT EYE */
	}


	bool updateRightEye(IplImage *img, IplImage *processedImg, CvRect *r, bool useDefaultSearchSpace, bool detailedOutput)
	{
		//store face dimensions in order to resize templates
		double newFaceWidth = r->width;
		double newFaceHeight = r->height;

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

		CvRect rightEyeSearchSpace;
		if(useDefaultSearchSpace == true)
		{
			rightEyeSearchSpace = this->getRightEyeSearchSpace(r);
		}
		else
		{
			rightEyeSearchSpace = this->getUpdatedSearchSpace(r,
				this->getRightEyeSearchSpace(r),this->rightEyeTpl,this->rightEyeLoc);
		}


		CvRect rightEyeLoc;
		//std::cout << "right eye=============================" << std::endl;
		bool rightEyeFound = getSearchSpace(img,processedImg,r,rightEyeTpl,rightEyeSearchSpace,rightEyeLoc,detailedOutput);
		//std::cout << "=============================right eye" << std::endl;

		if(rightEyeFound == false)
		{
			return false;
		}
		else //update right eye stuff
		{
			this->rightEyeTpl = rightEyeTpl;
			this->rightEyeLoc = rightEyeLoc;
			return true;
		}
		/* END RIGHT EYE */
	}

	
	/* Update locations of all features by running NCC's */
	bool updateFeatureLocs(IplImage *img, IplImage *processedImg, CvRect *r, bool detailedOutput)
	{
		//call isValidFace without cropping out templates
		//return this->isValidFace(img,processedImg,r,false);


		//store face dimensions in order to resize templates
		double newFaceWidth = r->width;
		double newFaceHeight = r->height;


		//update mouth coordinates

		//update face coordinates
		//are we updating twice?
		this->updateFaceCoords(r);


		//update eyebrows
		this->updateLeftEyebrow(img,processedImg,r,false,true);
		this->updateRightEyebrow(img,processedImg,r,false,true);
		
		//update eye subfeatures
		this->updateLeftEye(img,processedImg,r,false,true);
		this->updateRightEye(img,processedImg,r,false,true);

		//update mouth sub features
		bool updateMouthTopBottomSuccess = this->updateMouthTopBottom(img,processedImg,r);
		bool updateMouthLeftRightSuccess = this->updateMouthLeftRight(img,processedImg,r);
		//bool updateMouthTopBottomSuccess = true;
		
		//return (updateMouthTopBottomSuccess);

		this->detectEmotion();

		return true;
	}


	void detectEmotion()
	{
		double curBrowDistance = (this->rightEyebrowLoc.x - this->leftEyebrowLoc.x)/this->currentFaceWidth;
		double diffBrowDistance = (curBrowDistance - nBrowDistance) / nBrowDistance;
		
		double curLeftEyebrowRaised = (this->leftEyeLoc.y - this->leftEyebrowLoc.y)/this->currentFaceHeight;
		double curRightEyebrowRaised = (this->rightEyeLoc.y - this->rightEyebrowLoc.y)/this->currentFaceHeight;
		double diffLeftEyebrowRaised = (curLeftEyebrowRaised - nLeftEyebrowRaised) / nLeftEyebrowRaised;
		double diffRightEyebrowRaised = (curRightEyebrowRaised - nRightEyebrowRaised) / nRightEyebrowRaised;
		double diffEyebrowRaised = (diffLeftEyebrowRaised + diffRightEyebrowRaised) / 2.0;

		//double curMouthSmile = ((this->mouthLeftLoc.y + this->mouthRightLoc.y)/2.0)/this->currentFaceHeight;
		
		double curMouthSmile = (this->mouthTopLoc.y - ((this->mouthLeftLoc.y + this->mouthRightLoc.y)/2.0))/this->currentFaceHeight;
		double diffMouthSmile = (curMouthSmile - nMouthSmile) / nMouthSmile;

		double curMouthFrown = ((this->mouthLeftLoc.y + this->mouthRightLoc.y)/2.0 - this->mouthBottomLoc.y )/this->currentFaceHeight;
		double diffMouthFrown = (curMouthFrown - nMouthFrown) / nMouthFrown;

		double curMouthOpen = (double)(this->mouthBottomLoc.y - this->mouthTopLoc.y)/(double)this->currentFaceHeight;
		double diffMouthOpen = (double)(curMouthOpen - nMouthOpen) / (double)(nMouthOpen);
		
		
		 

		//std::cout << "browDistance: " << this->rightEyebrowLoc.x - this->leftEyebrowLoc.x << std::endl;
		//std::cout << "leftEyebrowRaised: " << this->leftEyeLoc.y - this->leftEyebrowLoc.y << std::endl;
		//std::cout << "rightEyebrowRaised: " << this->rightEyeLoc.y - this->rightEyebrowLoc.y << std::endl;
		//std::cout << "mouthOpen: " << this->mouthBottomLoc.y - this->mouthTopLoc.y << std::endl;
		
		
		
		//
		//std::cout << "mouthBot: " << this->mouthBottomLoc.y << std::endl;
		//std::cout << "mouthTop: " << this->mouthTopLoc.y << std::endl;


		
		

		std::cout << "mouthSmile: " << this->mouthTopLoc.y - (this->mouthLeftLoc.y + this->mouthRightLoc.y)/2 << std::endl;
		
		std::cout << "~~~~~~~~~~~~~~~~~~~~~~~~" << std::endl;
		//std::cout << "browDistance: " << this->rightEyebrowLoc.x - this->leftEyebrowLoc.x << std::endl;
		//std::cout << "diffBrowDistance: " << diffBrowDistance << std::endl;
		
		//std::cout << "diffLeftEyebrowRaised: " << diffLeftEyebrowRaised << std::endl;
		//std::cout << "diffRightEyebrowRaised: " << diffRightEyebrowRaised << std::endl;
		//std::cout << "diffEyebrowRaised: " << diffEyebrowRaised << std::endl;
		
		//std::cout << "nMouthSmile: " << nMouthSmile << std::endl;
		//std::cout << "curMouthSmile: " << curMouthSmile << std::endl;
		//std::cout << "diffMouthSmile: " << diffMouthSmile << std::endl;

		//std::cout << "nMouthFrown: " << nMouthFrown << std::endl;
		//std::cout << "curMouthFrown: " << curMouthFrown << std::endl;
		//std::cout << "diffMouthFrown: " << diffMouthFrown << std::endl;

		//std::cout << "nMouthOpen: " << nMouthOpen << std::endl;
		//std::cout << "curMouthOpen: " << curMouthOpen << std::endl;
		std::cout << "diffMouthOpen: " << diffMouthOpen << std::endl;
		
		//double emoThres = 0.02;//0.04;


		//horizontal distance btn eyebrows
		bool browClose = false;
		bool browFar = false;
		bool browHorizNeutral = false;
		double browDistThres = 0.10;
		if(diffBrowDistance < -browDistThres)
		{
			std::cout << "eyebrows are close together horizontally" << std::endl;
			browClose = true;
		}
		else if(diffBrowDistance > browDistThres)
		{
			std::cout << "eyebrows are far apart horizontally" << std::endl;
			browFar = true;
		}
		else
		{
			std::cout << "eyebrows are neutral horizontally" << std::endl;
			browHorizNeutral = true;
		}
		

		//eyebrows raised or lowered
		bool browLowered = false;
		bool browRaised = false;
		bool browRaiseNeutral = false;
		double browRaisedThres = 0.10;
		if(diffEyebrowRaised < -browRaisedThres)
		{
			std::cout << "eyebrows are lowered" << std::endl;
			browLowered = true;
		}
		else if(diffEyebrowRaised > browRaisedThres)
		{
			std::cout << "eyebrows are raised" << std::endl;
			browRaised = true;
		}
		else
		{
			std::cout << "eyebrows are neutral raised" << std::endl;
			browRaiseNeutral = true;
		}


		//smiling or not smiling
		bool smiling = false;
		bool notSmiling = false;
		bool smilingNeutral = false;
		double smileThres = 0.25;
		if(diffMouthSmile < -smileThres)
		{
			std::cout << "mouth smiling" << std::endl;
			smiling = true;
		}
		else if(diffMouthSmile > smileThres)
		{
			std::cout << "mouth not smiling" << std::endl;
			notSmiling = true;
		}
		else
		{
			std::cout << "mouth smile neutral" << std::endl;
			smilingNeutral = true;
		}

		//frowning or not frowning
		bool frowning = false;
		bool notFrowning = false;
		bool frowningNeutral = false;
		double frownThres = 0.15;
		if(diffMouthFrown < -frownThres)
		{
			std::cout << "mouth frowning" << std::endl;
			frowning = true;
		}
		else if(diffMouthFrown > frownThres)
		{
			std::cout << "mouth not frowning" << std::endl;
			notFrowning = true;
		}
		else
		{
			std::cout << "mouth frown neutral" << std::endl;
			frowningNeutral = true;
		}


		//mouth open or closed
		bool mouthOpen = false;
		bool mouthClosed = false;
		bool mouthOpenNeutral = false;
		double mouthOpenThres = 0.25;//0.15;//0.10;
		if(diffMouthOpen > mouthOpenThres)
		{
			std::cout << "mouth open" << std::endl;
			mouthOpen = true;
		}
		else if(diffMouthOpen < -mouthOpenThres)
		{
			std::cout << "mouth closed" << std::endl;
			mouthClosed = true;
		}
		else
		{
			std::cout << "mouth openness neutral" << std::endl;
			mouthOpenNeutral = true;
		}


		//neutral
		if(browHorizNeutral && browRaiseNeutral 
			&& smilingNeutral && frowningNeutral && mouthOpenNeutral)
		{
			std::cout << "neutral" << std::endl;
		}

		//angry
		if(browClose && browLowered && notSmiling)
		{
			std::cout << "angry" << std::endl;
			if(mouthOpen)
			{
				std::cout << "	mouth open" << std::endl;
			}
			if(frowning)
			{
				std::cout << "	frowning" << std::endl;
			}

		}

		//sad
		if( (browHorizNeutral || browFar) 
			&& (browRaiseNeutral || browRaised)
			&& notSmiling)
		{
			std::cout << "sad" << std::endl;
			if(mouthOpen)
			{
				std::cout << "	mouth open" << std::endl;
			}
			if(frowning)
			{
				std::cout << "	frowning" << std::endl;
			}
		}

		//afraid
		if( browClose && (browRaiseNeutral || browRaised) )
		{
			std::cout << "afraid" << std::endl;
			if(mouthOpen)
			{
				std::cout << "	mouth open" << std::endl;
			}
		}


		//surprised
		if((browHorizNeutral || browFar) && browRaised)
		{
			std::cout << "surprised" << std::endl;
			if(mouthOpen)
			{
				std::cout << "	mouth open" << std::endl;
			}
		}


		//disgusted
		if(browClose && browRaised && notSmiling)
		{
			std::cout << "disgusted" << std::endl;
			if(mouthOpen)
			{
				std::cout << "	mouth open" << std::endl;
			}
			if(frowning)
			{
				std::cout << "	frowning" << std::endl;
			}
		}

		//contempt
		if( (browHorizNeutral || browFar)
			&& (browRaiseNeutral || browLowered)
			&& (mouthOpenNeutral || mouthClosed)
			&& (smilingNeutral || notSmiling)
			)
		{
			std::cout << "contempt" << std::endl;
		}


		//happy
		if( (browHorizNeutral || browFar)
			&& (browRaiseNeutral || browRaised)
			&& smiling )
		{
			std::cout << "happy" << std::endl;
			if(mouthOpen)
			{
				std::cout << "	mouth open" << std::endl;
			}
		}

		//bored
		if( (browHorizNeutral || browClose)
			&& (browRaiseNeutral || browLowered)
			&& mouthOpenNeutral
			&& smilingNeutral
			)
		{
			std::cout << "bored" << std::endl;
		}


		//leftWink
		if( (diffLeftEyebrowRaised < -browRaisedThres)
			&& !(diffRightEyebrowRaised < -browRaisedThres)
			)
		{
			std::cout << "leftWink" << std::endl;
		}

		//rightWink
		else if( (diffRightEyebrowRaised < -browRaisedThres)
			&& !(diffLeftEyebrowRaised < -browRaisedThres)
			)
		{
			std::cout << "rightWink" << std::endl;
		}

		//skeptical
		else if( ((diffLeftEyebrowRaised > browRaisedThres) && !(diffRightEyebrowRaised > browRaisedThres))
			|| ((diffRightEyebrowRaised > browRaisedThres) && !(diffLeftEyebrowRaised > browRaisedThres))
			)
		{
			std::cout << "skeptical" << std::endl;
		}




		
	
		std::cout << "~~~~~~~~~~~~~~~~~~~~~~~~" << std::endl;
	}//detectEmotion()

	/* Return true if a face found by Haar is valid (has mouth,eyebrows,eyes) */
	bool isValidFace(IplImage *img, IplImage *processedImg, CvRect *r, bool doCrop = true, bool detailedOutput=false)
	{
		//update face coordinates
		this->updateFaceCoords(r);

		//store face dimensions in order to resize templates
		double newFaceWidth = r->width;
		double newFaceHeight = r->height;
		
		//return false if any of the features are not found

		/* MOUTH */
		if(this->updateMouth(img,processedImg,r,true) == false)
		{
			return false;
		}
		/* END MOUTH */

		/* LEFT EYEBROW */
		if(this->updateLeftEyebrow(img,processedImg,r,true,true) == false)
		{
			return false;
		}
		/* END LEFT EYEBROW */

		/* RIGHT EYEBROW */
		if(this->updateRightEyebrow(img,processedImg,r,true,true) == false)
		{
			return false;
		}
		/* END RIGHT EYEBROW */

		/* LEFT EYE */
		if(this->updateLeftEye(img,processedImg,r,true,true) == false)
		{
			return false;
		}
		/* END LEFT EYE */

		/* RIGHT EYE */
		if(this->updateRightEye(img,processedImg,r,true,true) == false)
		{
			return false;
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

		//crop out sub templates			

		/* MOUTH SUBFEATURES */
		//look at parent feature coords for search space
		CvRect parentLoc = this->mouthLoc;

		/* MOUTH TOP */
		CvRect topSearchSpace;
		topSearchSpace = this->getMouthTopSearchSpace();
		this->cropTemplate(img,topSearchSpace,this->mouthTopTpl);

		//yellow box
		rectangle(Mat(processedImg),Point(topSearchSpace.x,topSearchSpace.y),
			Point(topSearchSpace.x+topSearchSpace.width,topSearchSpace.y+topSearchSpace.height),
			CV_RGB(255,255,0),1,0,0);

		this->mouthTopLoc = topSearchSpace;
		/* END MOUTH TOP */

		/* MOUTH BOTTOM */
		CvRect bottomSearchSpace;
		bottomSearchSpace = this->getMouthBottomSearchSpace();
		this->cropTemplate(img,bottomSearchSpace,this->mouthBottomTpl);
		
		//pink box
		rectangle(Mat(processedImg),Point(bottomSearchSpace.x,bottomSearchSpace.y),
			Point(bottomSearchSpace.x+bottomSearchSpace.width,bottomSearchSpace.y+bottomSearchSpace.height),
			CV_RGB(255,0,255),1,0,0);

		this->mouthBottomLoc = bottomSearchSpace;
		/* END MOUTH BOTTOM */


		/* MOUTH LEFT */
		CvRect leftSearchSpace;
		leftSearchSpace = this->getMouthLeftSearchSpace();
		this->cropTemplate(img,leftSearchSpace,this->mouthLeftTpl);
		
		//pink box
		rectangle(Mat(processedImg),Point(leftSearchSpace.x,leftSearchSpace.y),
			Point(leftSearchSpace.x+leftSearchSpace.width,leftSearchSpace.y+leftSearchSpace.height),
			CV_RGB(255,0,255),1,0,0);

		this->mouthLeftLoc = leftSearchSpace;
		/* END MOUTH LEFT */


		/* MOUTH RIGHT */
		CvRect rightSearchSpace;
		rightSearchSpace = this->getMouthRightSearchSpace();
		this->cropTemplate(img,rightSearchSpace,this->mouthRightTpl);
		
		//pink box
		rectangle(Mat(processedImg),Point(rightSearchSpace.x,rightSearchSpace.y),
			Point(rightSearchSpace.x+rightSearchSpace.width,rightSearchSpace.y+rightSearchSpace.height),
			CV_RGB(255,0,255),1,0,0);

		this->mouthRightLoc = rightSearchSpace;
		/* END MOUTH RIGHT */
		/* END MOUTH SUBFEATURES */

		//store neutral positions
		//put this where new face is initialized                               
		nBrowDistance = (double)(this->rightEyebrowLoc.x - this->leftEyebrowLoc.x)/(double)this->currentFaceWidth;
		nLeftEyebrowRaised = (double)(this->leftEyeLoc.y - this->leftEyebrowLoc.y)/(double)this->currentFaceHeight;
		nRightEyebrowRaised = (double)(this->rightEyeLoc.y - this->rightEyebrowLoc.y)/(double)this->currentFaceHeight;
		
		nMouthOpen = (double)(this->mouthBottomLoc.y - this->mouthTopLoc.y)/(double)this->currentFaceHeight;
		
		std::cout << "****************" << std::endl;
		
		//nMouthOpen = (double)(this->mouthBottomLoc.y - this->mouthTopLoc.y);

		std::cout << "mouthBot: " <<  this->mouthBottomLoc.y << std::endl;
		std::cout << "mouthTop: " <<  this->mouthTopLoc.y << std::endl;

		std::cout << "****************" << std::endl;

		
		nMouthSmile = (this->mouthTopLoc.y - ((this->mouthLeftLoc.y + this->mouthRightLoc.y)/2.0)) / (double)this->currentFaceHeight;

		nMouthFrown = ((this->mouthLeftLoc.y + this->mouthRightLoc.y)/2.0 - this->mouthBottomLoc.y) / (double)this->currentFaceHeight;


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
