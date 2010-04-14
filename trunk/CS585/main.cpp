/**
 * Display video from webcam and detect faces
 */
#include <stdio.h>
//#include "cv.h"
#include <cv.h>
#include "highgui.h"
#include <iostream>

#include <vector>
#include "Face.h"

using namespace cv; //ADDED

CvHaarClassifierCascade *cascade;
CvMemStorage            *storage;


vector<Face*> oldFaces;
vector<Face*> newFaces;

void detectFaces( IplImage *img );

int main( int argc, char** argv )
{
	bool isVideo = true; //video
	//bool isVideo = false; //image

	//is image
	if(isVideo == false)
	{
		  //CvCapture *capture;
		  IplImage  *img;
		  char      *filename = "haarcascade_frontalface_default.xml";
		  //char      *filename = "haarcascade_frontalface_alt.xml";
		  //char      *imgfilename = "HowellFace50.jpg";
		  //char      *imgfilename = "grouppic2.jpg";
		  //char      *imgfilename = "ChrisFace.jpg";
		  char      *imgfilename = "templates/face.jpg";

		  cascade = ( CvHaarClassifierCascade* )cvLoad( filename, 0, 0, 0 );
		  storage = cvCreateMemStorage( 0 );
		  img     = cvLoadImage( imgfilename, 1 );

		  assert( cascade && storage && img );

		  cvNamedWindow( "video", 1 );
		  cvNamedWindow( "processed", 1 );

		  detectFaces( img );
		  cvWaitKey( 0 );

		  cvDestroyWindow( "video" );
		  cvDestroyWindow( "processed" );
		  
		  cvReleaseImage( &img );
		  cvReleaseHaarClassifierCascade( &cascade );
		  cvReleaseMemStorage( &storage );

		  return 0;
	}
	else //video
	{
		CvCapture *capture;
		IplImage  *frame;
		//IplImage  *processedFrame;

		int       key = ' ';
		char      *filename = "haarcascade_frontalface_default.xml";
		//char      *filename = "haarcascade_frontalface_alt.xml";
	 
		/* load the classifier
		   note that I put the file in the same directory with
		   this code */
		cascade = ( CvHaarClassifierCascade* )cvLoad( filename, 0, 0, 0 );
	 
		/* setup memory buffer; needed by the face detector */
		storage = cvCreateMemStorage( 0 );
	 
		/* initialize camera */
		capture = cvCaptureFromCAM( 0 );
	 
		/* always check */
		assert( cascade && storage && capture );
	 
		/* create a window */
		cvNamedWindow( "video", 1 );
		cvNamedWindow( "processed", 1 );
	 
		while( key != 'q' ) {
			// get a frame */
			frame = cvQueryFrame( capture );
	 
			// always check */
			if( !frame ) break;
	 
			// 'fix' frame */
			//cvFlip( frame, frame, -1 );
			frame->origin = 0;
			//processedFrame=cvCloneImage(frame);
	 
			// detect faces and display video */
			detectFaces( frame );
	 
			// quit if user press 'q' */
			key = cvWaitKey( 10 );
		}
	 
		// free memory */
		cvReleaseCapture( &capture );
		cvDestroyWindow( "video" );
		cvReleaseHaarClassifierCascade( &cascade );
		cvReleaseMemStorage( &storage );
	 
		return 0;
	}
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
 
boolean matchesOldFace(Point curTopLeftPoint, Face &matchedFace)
{
    double NOT_FOUND_VAL = 999.0;
    double SUM_THRESH = 0.2;
    
    double minSumDiff = NOT_FOUND_VAL;

	for(int i=0; i<(int)(oldFaces.size()); i++)
    {
        //L-distance btn top left coords of face and old face
		Point oldTopLeftPoint = ((Face*)(oldFaces.at(i)))->getTopLeftPoint();
        double currentXDiff = abs(curTopLeftPoint.x - oldTopLeftPoint.x);
        double currentYDiff = abs(curTopLeftPoint.y - oldTopLeftPoint.y);
        double currentSumDiff = currentXDiff + currentYDiff;
        
        if( currentSumDiff < SUM_THRESH 
            && currentSumDiff < minSumDiff)
        {
            minSumDiff = currentSumDiff;
			matchedFace = *((Face*)(oldFaces.at(i)));
        }
    }
    
    if(minSumDiff == NOT_FOUND_VAL)
    {
        return false;
    }
    else
    {
        return true;
    }
}


void detectFaces( IplImage *img )
{
    IplImage *processedImg;
	processedImg = cvCloneImage(img);

	int i;
 
    /* detect faces */
	//http://opencv.willowgarage.com/documentation/c/object_detection.html?highlight=cvhaardetectobjects
	// For a faster operation on real video images the settings are:
	//scale_factor =1.2, min_neighbors =2, flags = CV_HAAR_DO_CANNY_PRUNING ,
	//min_size = minimum possible face size (for example,  
	//1/4 to 1/16 of the image area in the case of video conferencing).
    CvSeq *faces = cvHaarDetectObjects(
            img,
            cascade,
            storage,
            1.2,//1.1,
            2,//3,
            CV_HAAR_DO_CANNY_PRUNING /*0*/ /*CV_HAAR_DO_CANNY_PRUNING*/, //pruning may speed up processing
            //cvSize( 40, 40 ) );
			cvSize( 50, 50 ) ); //30 by 30 good for grouppic.jpg
 
    /* for each face found, draw a red box */
	
	double oldFaceWidth = 228;
	double oldFaceHeight = 228;

    for( i = 0 ; i < ( faces ? faces->total : 0 ) ; i++ ) 
	{
	
	//performance estimation:
	for( int j=0; j<3; j++)
	{
        CvRect *r = ( CvRect* )cvGetSeqElem( faces, i );
	
		/*
		std::cout << "top left:" << r->x << "," << r->y << std::endl;
		std::cout << "bottom left:" << r->x + r->width << "," << r->y + r->height << std::endl;

		std::cout << "width: " << r->width << std::endl;
		std::cout << "height: " << r->height << std::endl;

		std::cout << "area: " << r->width * r->height << std::endl;
		*/

		double newFaceWidth = r->width;
		double newFaceHeight = r->height;

		Face *face = new Face(Point(r->x,r->y),Point(r->x+r->width,r->y+r->height));

		/* draw rectangle */
		//cvRectangle(img,
		//			cvPoint(minloc.x, minloc.y),
		//			cvPoint(minloc.x + tpl->width, minloc.y + tpl->height),
		//			CV_RGB(255, 0, 0), 1, 0, 0 );


		//std::cout << "max: " << "(" << maxloc.x << "," << maxloc.y << "): " << maxval << std::endl;

		//cvResetImageROI(img);
		if( face->isValidFace(img,processedImg,newFaceWidth,newFaceHeight) ) //above .6 reduces eyebrow noise a little
		{


			
			/*rectangle(Mat(processedImg),maxloc,Point(maxloc.x + tpl.cols, maxloc.y + tpl.rows),CV_RGB(0, 255, 0), 1, 0, 0 );
			std::cout << "max: " << "(" << maxloc.x << "," << maxloc.y << "): " << maxval << std::endl;*/



			//Face *currentFace = new Face();
			//currentFace->setLeftEye(tpl);


			cvResetImageROI(img);
			cvResetImageROI(processedImg);
			//
			//cvRectangle( processedImg,
			//	cvPoint( r->x, r->y ),
			//	cvPoint( r->x + r->width, r->y + r->height ),
			//	CV_RGB( 255, 0, 0 ), 1, 8, 0 );


						cvRectangle( processedImg,
				cvPoint( r->x, r->y ),
				cvPoint( r->x + r->width, r->y + r->height ),
				CV_RGB( 255, 0, 0 ), 1, 8, 0 );


		}
		else
		{
			cvResetImageROI(img);
			cvResetImageROI(processedImg);
		}
		
		
		//std::cout << "max: " << "(" << maxloc.x << "," << maxloc.y << "): " << maxval << std::endl;


			//write out image for debuging
			imwrite("image.jpg",Mat(processedImg));

	}//end 18 test performance evaluation
    
	}


 
    /* display video */
    cvShowImage( "video", img );
	cvShowImage( "processed", processedImg );
}
 