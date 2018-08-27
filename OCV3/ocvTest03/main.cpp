#include <stdio.h>
#include <iostream>
#include <opencv2/opencv.hpp>
#include "opencv2/core.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/xfeatures2d.hpp"
#include "opencv2/highgui.hpp"

using namespace cv;
using namespace cv::xfeatures2d;
using namespace std;

//当前img_1为原图像
//src未使用，dst为目标图像，graySrc为灰度图，normDst为归一化图像
Mat src, dst, img_1,graySrc,normDst;

//变量声明
int harrisThresh = 127;
int threshMax = 255;



//方法声明
void HarrisTrack(int, void *);



int main(int argc, char** argv) {
	//img_1 = imread("C:/Users/SANG-ASUS/Desktop/pic1.png");
	img_1 = imread("ppp.png");
	if (img_1.empty()) {
		printf("could not load image...\n");
		waitKey(0);
		return -1;
	}
	imshow("input image", img_1);

	
	//Harris角点检测 参考
	//https://blog.csdn.net/woxincd/article/details/60754658 

	//明天再写吧
	//cvCornerHarris(src, dst, 3);

	cvtColor(img_1, graySrc, CV_BGR2GRAY);
	imshow("gray", graySrc);

	//cornerHarris(src, dst, 3, 3, 1.0);
	
	namedWindow("harris", WINDOW_AUTOSIZE);
	createTrackbar("harrisTitle", "harris", &harrisThresh, threshMax, HarrisTrack);
	HarrisTrack(0,0);


	waitKey(0);
	return 0;
}

void HarrisTrack(int, void *)
{
	dst = Mat::zeros(graySrc.size(), CV_32FC1);
	//现在
	cornerHarris(graySrc, dst, 2, 3, 0.04);
	normalize(dst, normDst, 0,255, NORM_MINMAX);	
	//现在dst为ScaleAbs的结果
	convertScaleAbs(normDst, dst);

	Mat resultImg = img_1.clone();
	for (int row = 0; row  < resultImg.rows; row ++)
	{
		//获取到当前Row
		uchar* currentRow = dst.ptr(row);
		for (int col = 0; col < resultImg.cols; col++)
		{
			//索引到currentRow[0] 该数组的第一个数
			int value = (int)* currentRow;
			//大于阈值的才放出来
			if (value > harrisThresh)
			{
				//row col不要搞反了
				circle(resultImg, Point(col, row), 2, Scalar(0, 255, 0));
			}
			currentRow++;
		}
	}
	//展示resultImg处理结果
	imshow("harris", resultImg);

}
