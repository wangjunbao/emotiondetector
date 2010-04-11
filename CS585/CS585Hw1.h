#ifndef CS585HW1
#define CS585HW1

#include <cv.h>
#include "Image.h"
#include <cmath> //for absolute value
//using namespace std;

class CS585Hw1
{
public:
	CS585Hw1():
	  myChar(0)
	{
		//make a new window where we can show our processed results
		namedWindow("processed",1);
	}
	
	//Returns true if a color is skin-color
	bool isSkin(int red, int green, int blue)
	{
		//Compute color percentages for skin-color detection
		int totalRGB = red + green + blue;
		float percentR = (float)red / totalRGB;
		float percentG = (float)green / totalRGB;
		float percentB = (float)blue / totalRGB;

		//Detect skin-color pixels:
		//Based on my sampling: %R > %G > %B for skin-color
		//%R is never > 50% though, this also prevents
		//red objects from getting labeled as skin-color
		if( (percentR <= 0.50)
			&&(percentR > percentG)
			&& (percentG > percentB)
			)
		{
			return true;
		}
		return false;
	}

	
	//x histogram of skin-color
	void xSkinHisto(Image& img, int& leftX, int& rightX, float threshPercent, bool detailedOutput)
	{
		//allocate space to store the histogram
		//otherwise we may overflow the stack
		int* histogram;
		histogram = new int[img.getWidth()];
		
		Color c;
		for(int x=0; x<img.getWidth(); x++)
		{
			int skinCount = 0;
			for(int y=0; y<img.getHeight(); y++)
			{
				//Grab colors from current frame
				c = img.get(x,y);
				if(isSkin(c.r,c.g,c.b))
				{
					skinCount++;
				}
			}//end for y
			histogram[x] = skinCount;

			if(detailedOutput)
			{
				//draw the x-histogram along the bottom
				for(int hy=img.getHeight(); hy>(img.getHeight()-histogram[x]); hy--)
				{
					img.set(x,hy,0,255,0); //green
				}
			}
		}//end for x

		//find center of the face
		//the x with most skin-colored pixels
		int max = 0;
		for(int i=0; i<img.getWidth(); i++)
		{
			if(histogram[i] > histogram[max])
			{
				max = i;
			}
		}

		//find left bound of face
		int leftBound = max; //begin in the middle and go left
		while(
			leftBound > 0 && 
			((float)histogram[leftBound] / (float)histogram[max]) >= threshPercent
			)
		{
			if(!(
			leftBound > 0 && 
			((float)histogram[leftBound] / (float)histogram[max]) >= threshPercent
			))
			{
				break;
			}
			leftBound--;
		}

		//find right bound of face
		int rightBound = max; //begin in the middle and go right
		while(
			rightBound < img.getWidth()-1 && 
			((float)histogram[rightBound] / (float)histogram[max]) >= threshPercent
			)
		{
			if(!(
			rightBound < img.getWidth()-1 && 
			((float)histogram[rightBound] / (float)histogram[max]) >= threshPercent
			))
			{
				break;
			}
			rightBound++;
		}

		//return the x-coordinates for the face bounding box
		leftX = leftBound;
		rightX = rightBound;

		//cleanup!
		delete [] histogram;
	}

	
	//y histogram of skin-color
	void ySkinHisto(Image& img, int& topY, int& bottomY, float threshPercent, bool detailedOutput)
	{
		//allocate space to store the histogram
		//otherwise we may overflow the stack
		int* histogram;
		histogram = new int[img.getHeight()];
		
		Color c;
		for(int y=0; y<img.getHeight(); y++)
		{
			int skinCount = 0;
			for(int x=0; x<img.getWidth(); x++)
			{
				//Grab colors from current frame
				c = img.get(x,y);
				if(isSkin(c.r,c.g,c.b))
				{
					skinCount++;
				}
			}//end for x
			histogram[y] = skinCount;

			if(detailedOutput)
			{
				//draw the y-histogram along the left
				for(int hx=0; hx<histogram[y]; hx++)
				{
					img.set(hx,y,0,255,0); //green
				}
			}
		}//end for y

		//find center of the face
		//the y with most skin-colored pixels
		int max = 0;
		for(int i=0; i<img.getHeight(); i++)
		{
			if(histogram[i] > histogram[max])
			{
				max = i;
			}
		}

		//find top bound of face
		int topBound = max; //begin in the middle and go up
		while(
			topBound > 0 && 
			((float)histogram[topBound] / (float)histogram[max]) >= threshPercent
			)
		{
			if(!(
			topBound > 0 && 
			((float)histogram[topBound] / (float)histogram[max]) >= threshPercent
			))
			{
				break;
			}
			topBound--;
		}

		//find bottom bound of face
		int bottomBound = max; //begin in the middle and go down
		while(
			bottomBound < img.getHeight()-1 && 
			((float)histogram[bottomBound] / (float)histogram[max]) >= threshPercent
			)
		{
			if(!(
			bottomBound < img.getHeight()-1 && 
			((float)histogram[bottomBound] / (float)histogram[max]) >= threshPercent
			))
			{
				break;
			}
			bottomBound++;
		}

		//return the y-coordinates for the face bounding box
		topY = topBound;
		bottomY = bottomBound;

		//cleanup!
		delete [] histogram;
	}	
	
	//draw a box on an image given top-left and lower-right coordinates
	void drawBox(Image& img, const int leftX, const int topY, const int rightX, const int bottomY)
	{
		//draw the top line
		for(int topX = leftX; topX <= rightX; topX++)
		{
			img.set(topX,topY,0,255,0); //green
		}

		//draw the bottom line
		for(int bottomX = leftX; bottomX <= rightX; bottomX++)
		{
			img.set(bottomX,bottomY,0,255,0); //green
		}

		//draw the left line
		for(int leftY = topY; leftY <= bottomY; leftY++)
		{
			img.set(leftX,leftY,0,255,0); //green
		}

		//draw the right line
		for(int rightY = topY; rightY <= bottomY; rightY++)
		{
			img.set(rightX,rightY,0,255,0); //green
		}
	}

	
	//process the image, coloring in skin-color, motion, and detecting the face
	bool doWork(Mat& frame, Mat& prevFrame, bool detailedOutput)
	{
		processed(frame);
		
		//grab the image from the previous frame
		if(!prevFrame.empty())
		{
			prevProcessed(prevFrame);
		}
		//for the very first frame, there is no previous frame
		//so just use the current frame
		else
		{
			prevProcessed(frame);
		}
		
		//Variables for the current frame
		Color c;
		int red;
		int green;
		int blue;

		//Variables for motion detection
		Color prevC;
		int prevRed;
		int prevGreen;
		int prevBlue;
		int motionThreshold = 15;
		int diffRed;
		int diffGreen;
		int diffBlue;
		
		//Iterate over all pixels in the image
		//Check for skin-color and motion
		for(int x=0; x<processed.getWidth(); x++)
		{
			for(int y=0; y<processed.getHeight(); y++)
			{
				//Grab colors from current frame
				c = processed.get(x,y);
				red = c.r;
				green = c.g;
				blue = c.b;

				//Grab colors from previous frame
				prevC = prevProcessed.get(x,y);
				prevRed = prevC.r;
				prevGreen = prevC.g;
				prevBlue = prevC.b;

				//Difference btn current and previous colors
				diffRed = abs(red - prevRed);
				diffGreen = abs(green - prevGreen);
				diffBlue = abs(blue - prevBlue);

				//color non-skin pixels black
				//skin-color pixels remain their original color
				if(!isSkin(red,green,blue))
				{
					processed.set(x,y,0,0,0); //black
				}

				//color motion pixels red
				//pixels that are both skin-color and motion are also red
				//Check if the RGB values of the pixel differ from
				//the RGB values of that pixel in the previous frame
				//by a certain threshold
				else if(
					diffRed >= motionThreshold
					|| diffGreen >= motionThreshold
					|| diffBlue >= motionThreshold
					)	
				{
					processed.set(x,y,255,0,0); //red
				}

			}//end for y
		}//end for x


		//draw bounding box around the face:

		//coordinates of face bounding box
		int skinLeftX=0, skinTopY=0, skinRightX=0, skinBottomY=0;
		
		//center of face is where most skin-color pixels are
		//x's and y's that have percentThresh * (# skin pix in center of face)
		//are also counted as the face
		float percentThresh = 0.10f; 

		//get coordinates of face bounding box
		//histograms will be drawn if detailedOutput == true
		xSkinHisto(processed,skinLeftX,skinRightX,percentThresh,detailedOutput);
		ySkinHisto(processed,skinTopY,skinBottomY,percentThresh,detailedOutput);

		//draw bounding box around the face
		drawBox(processed,skinLeftX,skinTopY,skinRightX,skinBottomY);

		//show the processed image
		imshow("processed", processed.getImage());
		cvWaitKey(30);

		return true;
	}

	void setKey(char c)
	{
		myChar = c;
	}

	Image processed;
	Image prevProcessed; //the previous frame's image (for motion detection)
	char myChar;

};

#endif