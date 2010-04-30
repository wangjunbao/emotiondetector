#pragma once
#include <cv.h>
using namespace cv;

#include "Rect.h"
#include "Color.h"

/**************
 * Helper class which represents an image. The class contains
 * an pointer to a type of IplImage from the OpenCV imaging library.
 * This class allows simple access as well as some helper processing
 * operations.
 * 
 * !WARNING!
 * When creating an instance of the image class, you'll need to pass it
 * a pointer to an instance of the IplImage* class. Note that you should
 * NEVER pass in the exact IplImage* from a captured video. Instead copy
 * the video using cvCopy.
 *
 * Sam Epstein
 **************/
class Image
{
private:

	//Pointer to obect containing all the information
	Mat img;

public:
	
	//Constructs an empty instance
	Image(void);

	Image(int w, int h);

	~Image(void);

	void operator()(Mat& frame);

	//Raw data accessor
	Mat& getImage(){return img;}

	//Raw data modifier
	int getWidth(){return img.cols;}
	int getHeight(){return img.rows;}

	Color get(int x,int y);
	int getR(int x,int y);
	int getG(int x,int y);
	int getB(int x,int y);

	void set(int x,int y, Color &c);
	void set(int x,int y, int r, int g, int b);

	//Scales down the image by a factor represented by
	//scaleFactor
	void rescale(double scaleFactor);
};


