#pragma once
#include <cv.h>
using namespace cv;

class Face
{

public:

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

private:
	//templates:
	
	Mat face;
	
	Mat leftEye;


};
