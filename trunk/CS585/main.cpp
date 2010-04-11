/**
 * Display video from webcam and detect faces
 */
#include <stdio.h>
//#include "cv.h"
#include <cv.h>
#include "highgui.h"
#include <iostream>
using namespace cv; //ADDED

CvHaarClassifierCascade *cascade;
CvMemStorage            *storage;
 
void detectFaces( IplImage *img );

int main( int argc, char** argv )
{
	bool isVideo = true;
	//bool isVideo = false; //image

	//is image
	if(isVideo == false)
	{
		  //CvCapture *capture;
		  IplImage  *img;
		  //int       key;
		  //char      *filename = "haarcascade_frontalface_alt_tree.xml";
		  char      *filename = "haarcascade_frontalface_default.xml";
		  //char      *imgfilename = "HowellFace50.jpg";
		  //char      *imgfilename = "grouppic2.jpg";
		  char      *imgfilename = "ChrisFace.jpg";
		  //char      *imgfilename = "templates/face.jpg";

		  cascade = ( CvHaarClassifierCascade* )cvLoad( filename, 0, 0, 0 );
		  storage = cvCreateMemStorage( 0 );
		  //img     = cvLoadImage( argv[1], 1 );
		  img     = cvLoadImage( imgfilename, 1 );

		  assert( cascade && storage && img );

		  cvNamedWindow( "video", 1 );
		  detectFaces( img );
		  cvWaitKey( 0 );
		  cvDestroyWindow( "video" );
		  cvReleaseImage( &img );
		  cvReleaseHaarClassifierCascade( &cascade );
		  cvReleaseMemStorage( &storage );

		  return 0;
	}
	else //video
	{
		CvCapture *capture;
		IplImage  *frame;
		int       key = ' ';
		//char      *filename = "haarcascade_eye.xml";
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
	 
		while( key != 'q' ) {
			// get a frame */
			frame = cvQueryFrame( capture );
	 
			// always check */
			if( !frame ) break;
	 
			// 'fix' frame */
			//cvFlip( frame, frame, -1 );
			frame->origin = 0;
	 
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
	Mat *featureImg = new Mat;//();
	*featureImg = imread( "templates/"+filename, 1 );

	//Mat *resizedFeatureImg = new Mat;//();
	
	resize(*featureImg, resizedFeatureImg, Size((int)newFeatureWidth,(int)newFeatureHeight));
	//resize(*featureImg, *resizedFeatureImg, Size(newFeatureWidth,newFeatureHeight));
	//imwrite("resizedTemplates/"+filename,*resizedFeatureImg);
	//delete resizedFeatureImg;//(resizedFeatureImg);
	delete featureImg;

	//return resizedFeatureImg;
}
 
void detectFaces( IplImage *img )
{
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

    for( i = 0 ; i < ( faces ? faces->total : 0 ) ; i++ ) {
        CvRect *r = ( CvRect* )cvGetSeqElem( faces, i );
        
		/*
		cvRectangle( img,
                     cvPoint( r->x, r->y ),
                     cvPoint( r->x + r->width, r->y + r->height ),
                     CV_RGB( 255, 0, 0 ), 1, 8, 0 );
					 */

		CvFont font;
		double hScale=1.0;
		double vScale=1.0;
		int    lineWidth=1;
		cvInitFont(&font,CV_FONT_HERSHEY_SIMPLEX|CV_FONT_ITALIC, hScale,vScale,0,lineWidth);

		//cvPutText (img,"",cvPoint(100,100), &font, cvScalar(255,255,0));

		/*
		std::cout << "top left:" << r->x << "," << r->y << std::endl;
		std::cout << "bottom left:" << r->x + r->width << "," << r->y + r->height << std::endl;

		std::cout << "width: " << r->width << std::endl;
		std::cout << "height: " << r->height << std::endl;

		std::cout << "area: " << r->width * r->height << std::endl;
		*/

		double newFaceWidth = r->width;
		double newFaceHeight = r->height;

		//resizeFeatureTemplate("mouth.jpg",95,53,newFaceWidth,newFaceHeight);
		
		Mat tpl;
		resizeFeatureTemplate("lefteye.jpg",61,34,newFaceWidth,newFaceHeight,tpl);

		//eyes
		
		//cvMatchTemplate(

		//http://nashruddin.com/OpenCV_Region_of_Interest_(ROI)

		//IplImage *img = cvLoadImage("myphoto.jpg", 1);
		//IplImage *tpl = //cvLoadImage("eye.jpg", 1);
		
		

		 
		CvRect rect = cvRect(r->x, r->y, r->width/2, r->height/2);
		 
		cvSetImageROI(img, rect);
		 
		/*
		IplImage *res = cvCreateImage(cvSize(rect.width  - tpl->width  + 1,
											 rect.height - tpl->height + 1),
									  IPL_DEPTH_32F, 1);
									  */

		//Mat res(Size(rect.width - tpl.cols + 1, rect.height - tpl.rows + 1), IPL_DEPTH_32F, 1);
		Mat res;
		 
		/* perform template matching */
		//cvMatchTemplate(img, tpl, res, CV_TM_SQDIFF);
		matchTemplate(Mat(img), tpl, res, CV_TM_CCOEFF_NORMED);
		 
		/* find best matches location */
		//CvPoint    minloc, maxloc;
		Point minloc, maxloc;
		double    minval, maxval;

		//cvMinMaxLoc(res, &minval, &maxval, &minloc, &maxloc, 0);
		minMaxLoc(res, &minval, &maxval, &minloc, &maxloc); 

		/* draw rectangle */
		//cvRectangle(img,
		//			cvPoint(minloc.x, minloc.y),
		//			cvPoint(minloc.x + tpl->width, minloc.y + tpl->height),
		//			CV_RGB(255, 0, 0), 1, 0, 0 );



		//cvResetImageROI(img);
		if(maxval > 0.5)
		{

			
			rectangle(Mat(img),maxloc,Point(maxloc.x + tpl.cols, maxloc.y + tpl.rows),CV_RGB(0, 255, 0), 1, 0, 0 );
			std::cout << "max: " << "(" << maxloc.x << "," << maxloc.y << "): " << maxval << std::endl;

			cvResetImageROI(img);
			cvRectangle( img,
				cvPoint( r->x, r->y ),
				cvPoint( r->x + r->width, r->y + r->height ),
				CV_RGB( 255, 0, 0 ), 1, 8, 0 );
		}
		else
		{
			cvResetImageROI(img);
		}
		
		
		//std::cout << "max: " << "(" << maxloc.x << "," << maxloc.y << "): " << maxval << std::endl;


    }

	
 
    /* display video */
    cvShowImage( "video", img );
}
 