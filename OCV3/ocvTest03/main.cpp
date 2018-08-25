#include <stdio.h>
#include <iostream>
#include "opencv2/core.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/xfeatures2d.hpp"
#include "opencv2/highgui.hpp"

using namespace cv;
using namespace cv::xfeatures2d;
using namespace std;

int main(int argc, char** argv) {
	Mat img_1 = imread("C:/Users/SANG-ASUS/Desktop/pic1.png", IMREAD_GRAYSCALE);
	if (img_1.empty()) {
		printf("could not load image...\n");
		return -1;
	}
	imshow("input image", img_1);

	int minHessian = 400;
	Ptr<SURF> detector = SURF::create(minHessian);
	vector<KeyPoint> keypoints;
	detector->detect(img_1, keypoints);

	Mat img_keypoints1;
	drawKeypoints(img_1, keypoints, img_keypoints1, Scalar::all(-1), DrawMatchesFlags::DEFAULT);
	namedWindow("key points", CV_WINDOW_AUTOSIZE);
	imshow("key points", img_keypoints1);

	waitKey(0);
	return 0;
}