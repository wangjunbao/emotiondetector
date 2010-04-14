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
 
bool matchesOldFace(Point curTopLeftPoint, Face *matchedFace)
{
	std::cout << "\n===entering matchesOldFace===" << std::endl;

    double NOT_FOUND_VAL = 999.0;
    double SUM_THRESH = 0.2;
    
    double minSumDiff = NOT_FOUND_VAL;
	int matchedFaceIndex = -1;

	if(!oldFaces.empty())
	{

	for(int i=0; i<(int)(oldFaces.size()); i++)
    {
        //L-distance btn top left coords of face and old face
		//Point oldTopLeftPoint = ((Face*)(oldFaces.at(i)))->getTopLeftPoint();
		std::cout << "\n!!!before getting TopLeftPoint" << std::endl;
		
		//if(!oldFaces.at(i))
		//{
			//std::cout << "Top left is null! " << i << std::endl;
		//}
		//else
		//{
			Point oldTopLeftPoint = oldFaces.at(i)->getTopLeftPoint();
			std::cout << "Top Left Point: " << oldTopLeftPoint.x << "," << oldTopLeftPoint.y << std::endl;
		
		//Point oldTopLeftPoint = oldFaces.at(i)->getTopLeftPoint();
        //Point oldTopLeftPoint = Point(50,50);

		//Point oldTopLeftPoint = Point( ((Face*)(oldFaces.at(i)))->getR().x, ((Face*)(oldFaces.at(i)))->getR().y );
		//std::cout << "Top Left Point: " << oldTopLeftPoint.x << "," << oldTopLeftPoint.y << std::endl;
		std::cout << "\n!!!after getting TopLeftPoint" << std::endl;

		double currentXDiff = abs(curTopLeftPoint.x - oldTopLeftPoint.x);
        double currentYDiff = abs(curTopLeftPoint.y - oldTopLeftPoint.y);
        double currentSumDiff = currentXDiff + currentYDiff;

		std::cout << "currentSumDiff: " << currentSumDiff << std::endl;
        
        if( currentSumDiff < SUM_THRESH 
            && currentSumDiff < minSumDiff)
        {
            minSumDiff = currentSumDiff;
			matchedFace = oldFaces.at(i); //((Face*)(oldFaces.at(i)));
			matchedFaceIndex = i;
			std::cout << "i: " << i << std::endl;
        }

		//}//end else for null
    }

	}//end if
    
    if(minSumDiff == NOT_FOUND_VAL)
    {
        std::cout << "===leaving matchesOldFace===\n" << std::endl;

		return false;
    }
    else
    {
		std::cout << "deleting face" << std::endl;

		std::cout << "size before delete:" << oldFaces.size() <<std::endl;

		//delete the best matched face from oldFaces
		//to reduce search space for next call
		if(matchedFaceIndex >= 0 && matchedFaceIndex < (int)oldFaces.size())
		{
			oldFaces.erase(oldFaces.begin()+matchedFaceIndex);
		}

		std::cout << "size after delete:" << oldFaces.size() <<std::endl;

		std::cout << "===leaving matchesOldFace===\n" << std::endl;

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
	for( int j=0; j<1; j++)
	{
        CvRect *r = ( CvRect* )cvGetSeqElem( faces, i );

		Face *face = NULL;
		bool isOldFace = false;
		
		if(!oldFaces.empty())
		{
			std::cout << "\nold faces NOT empty " << oldFaces.size() << std::endl;
			
			if(oldFaces.at(0) == NULL)
			{
				std::cout << "Top Left Point is NULL" << std::endl;
			}
			else
			{
				std::cout << "Top Left Point: " << oldFaces.at(0)->getTopLeftPoint().x << "," << oldFaces.at(0)->getTopLeftPoint().y << std::endl;
			
				isOldFace = matchesOldFace(Point(r->x,r->y), face);
			}
		}
		else
		{
			std::cout << "\nold faces IS empty" << std::endl;
			//isOldFace = false;
		}

		if(isOldFace == false)
		{
			face = new Face(*r);
			if( face->isValidFace(img,processedImg,r) )
			{
				std::cout << "face IS valid" << std::endl;
				//update sub features

				//do Emotion Detection -> store it
                
                //add value to output

                //add face to newFaces
				std::cout << "ADDING from isOldFace == false" << std::endl;
				std::cout << "face top left: " << face->getTopLeftPoint().x <<"," << face->getTopLeftPoint().y << std::endl;
				newFaces.push_back(face);
			}
			else
			{
				std::cout << "face NOT valid" << std::endl;
				delete face;
			}
		}
		else if(isOldFace == true)
		{
			std::cout << "old face" << std::endl;

			//update sub features
			
			//do Emotion Detection -> store it
                
            //add value to output

			//add oldFace to newFaces
			
			std::cout << "ADDING from isOldFace == true" << std::endl;
			std::cout << "face top left: " << face->getTopLeftPoint().x <<"," << face->getTopLeftPoint().y << std::endl;
			newFaces.push_back(face);

			
		}
		//write out image for debuging
		//imwrite("image.jpg",Mat(processedImg));



	}//end 18 test performance evaluation
    
	}//end for faces

	//oldFaces.clear(); //remove remaining unfound faces from prev frame
	//oldFaces = newFaces; //store faces found in this frame for next frame


	std::cout << "oldFaces before swap: " << oldFaces.size() << std::endl;
	std::cout << "newFaces before swap: " << newFaces.size() << std::endl;


	oldFaces.swap(newFaces);
	
	std::cout << "oldFaces after swap: " << oldFaces.size() << std::endl;
	std::cout << "newFaces after swap: " << newFaces.size() << std::endl;

	
	newFaces.clear(); //new faces now contains old faces that were unfound

	std::cout << "oldFaces after clear: " << oldFaces.size() << std::endl;
	std::cout << "newFaces after clear: " << newFaces.size() << std::endl;
	std::cout << "\n" << std::endl;
 
    /* display video */
    cvShowImage( "video", img );
	cvShowImage( "processed", processedImg );
}
 