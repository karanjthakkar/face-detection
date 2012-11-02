/*************************************************************************************************
@file facedetect.cpp
@author Karan Thakkar
@brief Uses the Haar Cascade Classifier to detect face in a video feed and display it in a window
*************************************************************************************************/

#include "stdafx.h"
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include <iostream>

using namespace std;
using namespace cv;

/** Function Headers */
void detectAndDisplay(IplImage* frame );

/** Global variables */
CvHaarClassifierCascade* face_cascade = 0;	//Haar Classifier for face
CvMemStorage* pStorageface = 0;				// memory for detector to use
   
RNG rng(12345);

/**
 * @function main
 */
int main( int argc, const char** argv )
{
	CvCapture* capture;
	IplImage* frame = 0;
	int i=0;

	//Load the cascade
	face_cascade = (CvHaarClassifierCascade *)cvLoad("C:\\OpenCV-2.4.2\\opencv\\data\\haarcascades\\haarcascade_frontalface_alt_tree.xml", 0 , 0, 0);
	
	while (true)
    {
		//Read the video stream
		capture = cvCaptureFromCAM(1);	
		frame = cvQueryFrame( capture );
  
		//-- 3. Apply the classifier to the frame
		detectAndDisplay(frame);

	    int c = waitKey(10);
	    if( (char)c == 27 ) { exit(0); } 

    }

    // clean up and release resources
    cvReleaseImage(&frame);
	if(face_cascade) cvReleaseHaarClassifierCascade(&face_cascade);
	if(pStorageface) cvReleaseMemStorage(&pStorageface);

	return 0;

}

/**
 * @function detectAndDisplay
 */
void detectAndDisplay(IplImage* frame )
{
	CvSeq * pFaceRectSeq;               // memory-access interface
	pStorageface = cvCreateMemStorage(0);
	cvClearMemStorage(pStorageface);

    // detect faces in image
	pFaceRectSeq = cvHaarDetectObjects
		(frame, face_cascade, pStorageface,
		1.1,                        // increase search scale by 10% each pass
		3,                          // merge groups of three detections
		0,						    // 0(entire region) or CV_HAAR_DO_CANNY_PRUNING(skip regions unlikely to contain a face)
		cvSize(40, 40)  			// smallest size face to detect = 40x40
		);            

	const char * DISPLAY_WINDOW = "Haar Window";
	int i;
	
	// create a window to display detected faces
	cvNamedWindow(DISPLAY_WINDOW, CV_WINDOW_AUTOSIZE);

	// draw a rectangular outline around each detection
	for(i=0;i<(pFaceRectSeq? pFaceRectSeq->total:0); i++ )
	{
		CvRect* r = (CvRect*)cvGetSeqElem(pFaceRectSeq, i);
		CvPoint pt1 = { r->x, r->y };
		CvPoint pt2 = { r->x + r->width, r->y + r->height };
		cvRectangle(frame, pt1, pt2, CV_RGB(255,255,255), 3, 4, 0);
	}

	// display face detections
	cvShowImage(DISPLAY_WINDOW, frame);
}

         