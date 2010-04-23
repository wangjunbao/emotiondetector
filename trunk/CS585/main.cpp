/**
 * Display video from webcam and detect faces
 */
#include <stdio.h>
#include <cv.h>
#include "highgui.h"
#include <iostream>

#include <vector>
#include "Face.h"

using namespace cv;

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
		  //char      *imgfilename = "ChrisFace.jpg"; //this does not work well now but face may be too big
		  char      *imgfilename = "ChrisFace50.jpg";
		  //char      *imgfilename = "averageFace.jpg";
		  //char      *imgfilename = "templates/face.jpg";
		  //char      *imgfilename = "templates - Howell/face.jpg";

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
		VideoCapture cap(0); // open the default camera
		if(!cap.isOpened())  // check if we succeeded
			return -1;
		
		Mat frame;

		int       key = ' ';
		char      *filename = "haarcascade_frontalface_default.xml";

		///* load the classifier
		//   note that I put the file in the same directory with
		//   this code */
		cascade = ( CvHaarClassifierCascade* )cvLoad( filename, 0, 0, 0 );
	 
		///* setup memory buffer; needed by the face detector */
		storage = cvCreateMemStorage( 0 );
	 
		/* create a window */
		cvNamedWindow( "video", 1 );
		cvNamedWindow( "processed", 1 );
	 
		int frameCount = 0;
		while( key != 'q' ) {
			
			frameCount++;
			//std::cout << frameCount << ": ";

			// get a new frame from camera
			cap >> frame; 
	 
			// detect faces and display video */
			detectFaces( &IplImage(frame) );

			// quit if user press 'q' */
			key = cvWaitKey( 10 );
		}
	 
		//// free memory */
		cvDestroyWindow( "video" );
		cvDestroyWindow( "processed" );
		cvReleaseHaarClassifierCascade( &cascade );
		cvReleaseMemStorage( &storage );

		return 0;
	}//end else
}


/* Checks if a face found in current frame is contained inside an existing face */
bool containedInOldFace(int faceX, int faceY, int faceWidth, int faceHeight)
{
	//bottom right points of face to check
	int faceBottomX = faceX + faceWidth;
	int faceBottomY = faceY + faceHeight;

	//compare with faces found in previous frame(oldFaces) and in this frame (newFaces) 

	//check matched old faces
	for(int i=0; i<(int)(oldFaces.size()); i++)
    {
		Face *oldFace = oldFaces.at(i);
		int oldFaceWidth = oldFace->getWidth();
		int oldFaceHeight = oldFace->getHeight();
		int oldFaceX = oldFace->getTopLeftPoint().x;
		int oldFaceY = oldFace->getTopLeftPoint().y;
		int oldFaceBottomX = oldFaceX + oldFaceWidth;
		int oldFaceBottomY = oldFaceY + oldFaceHeight;

		//first check if the face is smaller than the old face
		//b/c it could be the case that small fake face was found first before a large real face
		if( (faceWidth < oldFaceWidth) && (faceHeight < oldFaceHeight) )
		{
			//top left corner is contained in old face
			if( (faceX > oldFaceX) && (faceX < oldFaceBottomX) 
				&& (faceY > oldFaceY) && (faceY < oldFaceBottomY)
				)
			{
				return true;
			}
			//bottom right corner is contained in old face
			else if( (faceBottomX > oldFaceX) && (faceBottomX < oldFaceBottomX) 
					 && (faceBottomY > oldFaceY) && (faceBottomY < oldFaceBottomY)
				    )
			{
				return true;
			}
		}
	}//end for

	//check against new faces vector (this includes matched oldFaces)
    for(int i=0; i<(int)(newFaces.size()); i++)
	{
		Face *oldFace = newFaces.at(i);
		int oldFaceWidth = oldFace->getWidth();
		int oldFaceHeight = oldFace->getHeight();
		int oldFaceX = oldFace->getTopLeftPoint().x;
		int oldFaceY = oldFace->getTopLeftPoint().y;
		int oldFaceBottomX = oldFaceX + oldFaceWidth;
		int oldFaceBottomY = oldFaceY + oldFaceHeight;

		//first check if the face is smaller than the old face
		//b/c it could be the case that small fake face was found first before a large real face
		if( (faceWidth < oldFaceWidth) && (faceHeight < oldFaceHeight) )
		{
			//top left corner is contained in old face
			if( (faceX > oldFaceX) && (faceX < oldFaceBottomX) 
				&& (faceY > oldFaceY) && (faceY < oldFaceBottomY)
				)
			{
				return true;
			}
			//bottom right corner is contained in old face
			else if( (faceBottomX > oldFaceX) && (faceBottomX < oldFaceBottomX) 
					 && (faceBottomY > oldFaceY) && (faceBottomY < oldFaceBottomY)
				    )
			{
				return true;
			}
		}
	}//end for

	return false;
}//end containedInOldFace


/* Checks if a face found in current frame matches an existing face */
bool matchesOldFace(Point curTopLeftPoint, int width, int height, Face* matchedFace)
{
    double NOT_FOUND_VAL = 999.0;
	double SUM_THRESH = 0.08; //calc: 2/50 + 2/50 = 0.08 
    
    double minSumDiff = NOT_FOUND_VAL;
	int matchedFaceIndex = -1;

	//loop will not execute if oldFaces is empty and false will be returned
	for(int i=0; i<(int)(oldFaces.size()); i++)
    {
		assert (oldFaces.at(i)); //face from oldFaces should not be null
		Point oldTopLeftPoint = oldFaces.at(i)->getTopLeftPoint();

		//L-distance percentages btn top left coords of face and old face
		double currentXDiff = (double)abs(curTopLeftPoint.x - oldTopLeftPoint.x) / (double)width;
        double currentYDiff = (double)abs(curTopLeftPoint.y - oldTopLeftPoint.y) / (double)height;
        double currentSumDiff = currentXDiff + currentYDiff;

		//std::cout << "currentSumDiff: " << currentSumDiff << std::endl;
        
        if( currentSumDiff < SUM_THRESH 
            && currentSumDiff < minSumDiff)
        {
            minSumDiff = currentSumDiff;
			*matchedFace = *oldFaces.at(i); //now matchedFace and oldFace[i] point two distinct objects of same value
			matchedFaceIndex = i;
        }
    }
    
    if(minSumDiff == NOT_FOUND_VAL)
    {
		return false;
    }
    else
    {
		//delete the best matched face from oldFaces
		//to reduce search space for next call
		if(matchedFaceIndex >= 0 && matchedFaceIndex < (int)oldFaces.size())
		{
			delete oldFaces.at(matchedFaceIndex); //delete actual object			
			oldFaces.erase(oldFaces.begin()+matchedFaceIndex); //delete pointer to object from oldFaces vector
		}

        return true;
    }
}//end matchesOldFace

/* Find all possible face objects in the current frame */
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
            CV_HAAR_DO_CANNY_PRUNING /*0*/, //pruning may speed up processing
            //cvSize( 40, 40 ) );
			cvSize( 50, 50 ) ); //30 by 30 good for grouppic.jpg
			//cvSize( 75, 75 ) );

    for( i = 0 ; i < ( faces ? faces->total : 0 ) ; i++ ) 
	{
        CvRect *r = ( CvRect* )cvGetSeqElem( faces, i );

		//std::cout << "r: " << r->width << " by " << r->height << std::endl;

		//white box for all faces found by Haar but not necessarily true faces
		cvRectangle( processedImg,
			cvPoint( r->x, r->y ),
			cvPoint( r->x + r->width, r->y + r->height ),
			CV_RGB( 255, 255, 255 ), 1, 8, 0 );

		//create a new face object
		//if it is found to match an old face, it will take on the old face's values
		Face *face = new Face(*r);

		//try to match this face with an old face
		bool isOldFace = matchesOldFace(Point(r->x,r->y), r->width, r->height, face);

		if(isOldFace == false)
		{
			//check if the face is contained within an existing old face
			if(containedInOldFace(r->x,r->y,r->width,r->height) == true)
			{
				//std::cout << "face contained in old face" << std::endl;
		
				//purple box for all faces found by Haar but not necessarily true faces
				cvRectangle( processedImg,
					cvPoint( r->x, r->y ),
					cvPoint( r->x + r->width, r->y + r->height ),
					CV_RGB( 128, 0, 255 ), 1, 8, 0 );
				
				delete face;
			}
			else if( face->isValidFace(img,processedImg,r) )
			{
				//std::cout << "new valid face at: (" << face->getTopLeftPoint().x <<"," << face->getTopLeftPoint().y << ")" << std::endl;

				//update sub features

				//do Emotion Detection -> store it
                
				//add value to output

				//add face to newFaces
				newFaces.push_back(face);
			}
			else //face is not valid, delete the object
			{
				delete face;
			}
		}
		else if(isOldFace == true)
		{
			//std::cout << "old face matched at: (" << face->getTopLeftPoint().x << "," << face->getTopLeftPoint().y << ")" << std::endl;

			//update sub features
			bool updateMouthSuccess = face->updateMouthSubFeatureLocs(img,processedImg,*r);
			
			if(updateMouthSuccess == false)
			{
				delete face;
			}
			else
			{
				face->drawBox(img,processedImg,r);	//draw a yellow box on faces matched with old faces
				newFaces.push_back(face);			//add oldFace to newFaces
				//do Emotion Detection -> store it
                //add value to output
			}
			
			//for debugging:
			//face->isValidFace(img,processedImg,r); //DELETE
		}
		
	}//end for faces

	//newFaces vector becomes oldFaces vector for next frame
	oldFaces.swap(newFaces);
	
	//delete all the unmatched old faces (now in newFaces vector):
	for(int i=0; i<(int)newFaces.size(); i++)	//delete all the face objects first
	{
		delete newFaces.at(i);
	}

	newFaces.clear();	//delete all the face pointers from the vector
 
    /* display video */
    cvShowImage( "video", img );
	cvShowImage( "processed", processedImg );

	//free memory
	cvReleaseImage( &processedImg );
}
 