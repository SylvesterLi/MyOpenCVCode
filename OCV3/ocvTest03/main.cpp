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
//最大角点数
int maxCorner = 5;


//方法声明
void HarrisTrack(int, void *);
void ShiTomasiTrack(int, void *);


int main(int argc, char** argv) {
	src = imread("C:/Users/SANG-ASUS/Desktop/pic1.png");
	//img_1 = imread("ppp.png");
	if (src.empty()) {
		printf("could not load image...\n");
		waitKey(0);
		return -1;
	}
	imshow("raw image", src);

	cvtColor(src, graySrc, CV_BGR2GRAY);
	imshow("gray", graySrc);
	//这里展示了两个窗口，一个是原图，一个是灰度图


	#pragma region Harris 角点检测

	//Harris角点检测 参考
	//https://blog.csdn.net/woxincd/article/details/60754658 

	//namedWindow("harris", WINDOW_AUTOSIZE);
	//createTrackbar("harrisTitle", "harris", &harrisThresh, threshMax, HarrisTrack);
	//HarrisTrack(0,0);  

	#pragma endregion

	#pragma region ShiTomasi 角点检测 

	/*

	namedWindow("Good", WINDOW_AUTOSIZE);
	//ShiTomasi角点检测  
	createTrackbar("trackBarName", "Good", &maxCorner, 255, ShiTomasiTrack);
	ShiTomasiTrack(0, 0);
	
	*/

	#pragma endregion

	#pragma region 自定义角点检测

	//参考文章：
	//https://blog.csdn.net/weixin_41695564/article/details/79979784  

	


	#pragma endregion

	#pragma region SURF 特征检测

	/*

	//Hessian有点像阈值，值越大 特征点越多
	int minHessisan = 400;
	//现在创建检测器
	Ptr<SURF> detector = SURF::create(minHessisan);
	vector<KeyPoint> keypoints;//存到这来
	//检测
	detector->detect(src, keypoints);
	Mat kpImage;
	//绘制关键点
	drawKeypoints(src, keypoints, kpImage);

	namedWindow("result", WINDOW_AUTOSIZE);
	imshow("result", kpImage);

	*/

	#pragma endregion


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

void ShiTomasiTrack(int, void *)
{
	if (maxCorner < 1) { maxCorner = 1; }
	//最小可接受向量值  不知道是啥
	double qualityLv = 0.01;
	//两个角点最小间隔 防止同一角点被多次标记
	double minValue = 10;
	//保存Corners的数组 dst用于draw
	vector<Point2f> corners;

	//需要注意，这里并未使用Src，用的是img_1
	dst = img_1.clone();
	goodFeaturesToTrack(graySrc, corners, maxCorner, qualityLv, minValue);//后面保持默认

	//Draw Corners
	for (int i = 0;  i < corners.size(); i++)
	{
		circle(dst, corners[i], 2, Scalar(0, 255, 0));
	}
	imshow("Good", dst);


}
