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

//��ǰimg_1Ϊԭͼ��
//srcδʹ�ã�dstΪĿ��ͼ��graySrcΪ�Ҷ�ͼ��normDstΪ��һ��ͼ��
Mat src, dst, img_1,graySrc,normDst;

//��������
int harrisThresh = 127;
int threshMax = 255;
//���ǵ���
int maxCorner = 5;


//��������
void HarrisTrack(int, void *);
void ShiTomasiTrack(int, void *);


int main(int argc, char** argv) {
	src = imread("C:/Users/SANG-Surface/Desktop/peo.jpg");
	//img_1 = imread("ppp.png");
	if (src.empty()) {
		printf("could not load image...\n");
		waitKey(0);
		return -1;
	}
	imshow("raw image", src);

	cvtColor(src, graySrc, CV_BGR2GRAY);
	imshow("gray", graySrc);
	//����չʾ���������ڣ�һ����ԭͼ��һ���ǻҶ�ͼ


	#pragma region Harris �ǵ���

	//Harris�ǵ��� �ο�
	//https://blog.csdn.net/woxincd/article/details/60754658 

	//namedWindow("harris", WINDOW_AUTOSIZE);
	//createTrackbar("harrisTitle", "harris", &harrisThresh, threshMax, HarrisTrack);
	//HarrisTrack(0,0);  

	#pragma endregion

	#pragma region ShiTomasi �ǵ��� 

	/*

	namedWindow("Good", WINDOW_AUTOSIZE);
	//ShiTomasi�ǵ���  
	createTrackbar("trackBarName", "Good", &maxCorner, 255, ShiTomasiTrack);
	ShiTomasiTrack(0, 0);
	
	*/

	#pragma endregion

	#pragma region �Զ���ǵ���

	//�ο����£�
	//https://blog.csdn.net/weixin_41695564/article/details/79979784  

	


	#pragma endregion

	#pragma region SURF �������

	/*

	//Hessian�е�����ֵ��ֵԽ�� ������Խ��
	int minHessisan = 400;
	//���ڴ��������
	Ptr<SURF> detector = SURF::create(minHessisan);
	vector<KeyPoint> keypoints;//�浽����
	//���
	detector->detect(src, keypoints);
	Mat kpImage;
	//���ƹؼ���
	drawKeypoints(src, keypoints, kpImage);

	namedWindow("result", WINDOW_AUTOSIZE);
	imshow("result", kpImage);

	*/

	#pragma endregion

	#pragma region SIFT �������
	
	/*
	//SIFT��SURF������һģһ����

	//numOfFeaturesָ����������ĸ���
	int numOfFeatures = 400;
	//���ڴ��������
	Ptr<SIFT> detector = SIFT::create(numOfFeatures);
	vector<KeyPoint> keypoints;//�浽����
	//��� 
	detector->detect(src, keypoints);
	Mat kpImage;
	//���ƹؼ���
	drawKeypoints(src, keypoints, kpImage);

	namedWindow("result", WINDOW_AUTOSIZE);
	imshow("result", kpImage);
	*/

	#pragma endregion

	#pragma region Descriptors������

	/*
	//detector winSizae blockSize blockStrike cellSize bins(9������)
	HOGDescriptor detector(Size(64, 128), Size(16, 16), Size(8, 8), Size(8, 8), 9);
	//����� ���� ������ �����descriptors�У����������ӵ�λ�÷���Points��Location��
	vector<Point> locations;
	vector<float> descriptors;
	detector.compute(graySrc, descriptors, Size(0, 0), Size(0, 0), locations);
	printf("%d", descriptors.size());
	waitKey(0);
	*/

	#pragma endregion

	#pragma region HOG+SVM ��Ⱥ���
	
	/*
	//SVM �����Ⱥ 7938000�������� �ٶȽ���
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

	


	waitKey(0);
	return 0;
}

void HarrisTrack(int, void *)
{
	dst = Mat::zeros(graySrc.size(), CV_32FC1);
	//����
	cornerHarris(graySrc, dst, 2, 3, 0.04);
	normalize(dst, normDst, 0,255, NORM_MINMAX);	
	//����dstΪScaleAbs�Ľ��
	convertScaleAbs(normDst, dst);

	Mat resultImg = img_1.clone();
	for (int row = 0; row  < resultImg.rows; row ++)
	{
		//��ȡ����ǰRow
		uchar* currentRow = dst.ptr(row);
		for (int col = 0; col < resultImg.cols; col++)
		{
			//������currentRow[0] ������ĵ�һ����
			int value = (int)* currentRow;
			//������ֵ�Ĳŷų���
			if (value > harrisThresh)
			{
				//row col��Ҫ�㷴��
				circle(resultImg, Point(col, row), 2, Scalar(0, 255, 0));
			}
			currentRow++;
		}
	}
	//չʾresultImg������
	imshow("harris", resultImg);

}

void ShiTomasiTrack(int, void *)
{
	if (maxCorner < 1) { maxCorner = 1; }
	//��С�ɽ�������ֵ  ��֪����ɶ
	double qualityLv = 0.01;
	//�����ǵ���С��� ��ֹͬһ�ǵ㱻��α��
	double minValue = 10;
	//����Corners������ dst����draw
	vector<Point2f> corners;

	//��Ҫע�⣬���ﲢδʹ��Src���õ���img_1
	dst = img_1.clone();
	goodFeaturesToTrack(graySrc, corners, maxCorner, qualityLv, minValue);//���汣��Ĭ��

	//Draw Corners
	for (int i = 0;  i < corners.size(); i++)
	{
		circle(dst, corners[i], 2, Scalar(0, 255, 0));
	}
	imshow("Good", dst);


}
