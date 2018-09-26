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
	src = imread("C:/Users/SANG-Surface/Desktop/peo.jpg");
	img_1 = imread("C:/Users/SANG-Surface/Desktop/peoPart.png");
	//img_1 = imread("ppp.png");
	if (src.empty()) {
		printf("could not load image...\n");
		waitKey(0);
		return -1;
	}
	imshow("raw image", src);
	imshow("Part Image", img_1);
	//cvtColor(src, graySrc, CV_BGR2GRAY);
	//imshow("gray", graySrc);
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

	#pragma region SIFT 特征检测
	
	/*
	//SIFT跟SURF代码是一模一样的

	//numOfFeatures指的是特征点的个数
	int numOfFeatures = 400;
	//现在创建检测器
	Ptr<SIFT> detector = SIFT::create(numOfFeatures);
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

	#pragma region Descriptors描述子

	/*
	//detector winSizae blockSize blockStrike cellSize bins(9个向量)
	HOGDescriptor detector(Size(64, 128), Size(16, 16), Size(8, 8), Size(8, 8), 9);
	//检测器 检测出 描述子 存放在descriptors中，其中描述子的位置放在Points的Location中
	vector<Point> locations;
	vector<float> descriptors;
	detector.compute(graySrc, descriptors, Size(0, 0), Size(0, 0), locations);
	printf("%d", descriptors.size());
	waitKey(0);
	*/

	#pragma endregion

	#pragma region HOG+SVM 人群检测
	
	/*
	//SVM 检测人群 7938000个描述子 速度较慢
	HOGDescriptor hog = HOGDescriptor();
	hog.setSVMDetector(hog.getDefaultPeopleDetector());
	vector<Rect> foLocations;
	hog.detectMultiScale(src, foLocations,0, Size(8, 8), Size(32, 32), 1.05);
	Mat newSrc = src.clone();
	for (size_t i = 0; i < foLocations.size(); i++)
	{
		rectangle(newSrc, foLocations[i], Scalar(0, 255, 0),3);

	}
	imshow("hhh", newSrc);
	*/

	#pragma endregion

	#pragma region Descriptor 描述子
	/*
	//作用：匹配两张图像
	
	//需要两个描述子
	//本次采用SURF描述子
	Ptr<SURF> detector = SURF::create(400);
	//储存两个描述子的keypoint
	vector<KeyPoint> keyPoint_1;
	vector<KeyPoint> keyPoint_2;

	//声明两个描述子
	Mat descriptor_1, descriptor_2;
	detector->detectAndCompute(src, Mat(), keyPoint_1, descriptor_1);
	detector->detectAndCompute(img_1, Mat(), keyPoint_2, descriptor_2);

	//匹配
	BFMatcher bfMatcher;
	vector<DMatch> matches;
	bfMatcher.match(descriptor_1, descriptor_2, matches);

	//绘画
	Mat resImg;
	drawMatches(src, keyPoint_1, img_1, keyPoint_2, matches, resImg);
	imshow("res", resImg);

	*/

	#pragma endregion

	//作用：匹配两张图像
	
	//需要两个描述子
	//本次采用SURF描述子
	Ptr<SURF> detector = SURF::create(400);
	//储存两个描述子的keypoint
	vector<KeyPoint> keyPoint_1;
	vector<KeyPoint> keyPoint_2;

	//声明两个描述子
	Mat descriptor_1, descriptor_2;
	detector->detectAndCompute(src, Mat(), keyPoint_1, descriptor_1);
	detector->detectAndCompute(img_1, Mat(), keyPoint_2, descriptor_2);

	//匹配
	FlannBasedMatcher flannMatches;
	vector<DMatch> matches;//匹配了放这儿
	flannMatches.match(descriptor_1, descriptor_2, matches);

	//找到好的匹配点filter good matches point 过滤
	//不过滤跟BFMatch效果一样，都是花的
	double minDist = 1000;
	double maxDist = 0;
	for (int i = 0; i < descriptor_1.rows; i++)
	{
		double dist = matches[i].distance;
		if (dist > maxDist)
		{
			maxDist = dist;
		}
		if (dist < minDist)
		{
			minDist = dist;
		}
		printf("max:%f\n", maxDist);
		printf("min:%f\n", minDist);
	}//这儿算是划定最大最小值，下面才是有点像归一化
	printf("max:%f\n", maxDist);
	printf("min:%f\n", minDist);

	vector<DMatch> goodMatch;
	for (int j = 0; j < descriptor_1.rows; j++)
	{
		double dist = matches[j].distance;
		if (dist < max(2 * minDist, 0.02))//这个是干嘛的？
		{
			goodMatch.push_back(matches[j]);
		}
	}



	Mat resImg;
	drawMatches(src, keyPoint_1, img_1, keyPoint_2, goodMatch, resImg);
	imshow("res img", resImg);

	


	waitKey(0);
	return 0;
}

void HarrisTrack(int, void *)
{
	dst = Mat::zeros(graySrc.size(), CV_32FC1);
	
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
