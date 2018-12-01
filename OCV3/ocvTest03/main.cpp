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
int contours_TH = 127;
int skew_TH = 25;


//��������
void HarrisTrack(int, void *);
void ShiTomasiTrack(int, void *);
void findROI(int, void *);//Ѱ�ұ�Ե
void Check_Skew();



int main(int argc, char** argv) {
	//������Ҫע�⣺��Surface����SANG-Surface
	src = imread("C:/Users/SANG-Surface/Desktop/magz.jpg");
	//img_1 = imread("C:/Users/SANG-Surface/Desktop/233.png");
	//img_1 = imread("ppp.png");
	if (src.empty()) {
		printf("could not load image...\n");
		waitKey(0);
		return -1;
	}
	imshow("raw image", src);
	//imshow("Part Image", img_1);
	//cvtColor(src, graySrc, CV_BGR2GRAY);
	//imshow("gray", graySrc);
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

	#pragma region Descriptor ������
	/*
	//���ã�ƥ������ͼ��
	
	//��Ҫ����������
	//���β���SURF������
	Ptr<SURF> detector = SURF::create(400);
	//�������������ӵ�keypoint
	vector<KeyPoint> keyPoint_1;
	vector<KeyPoint> keyPoint_2;

	//��������������
	Mat descriptor_1, descriptor_2;
	detector->detectAndCompute(src, Mat(), keyPoint_1, descriptor_1);
	detector->detectAndCompute(img_1, Mat(), keyPoint_2, descriptor_2);

	//ƥ��
	BFMatcher bfMatcher;
	vector<DMatch> matches;
	bfMatcher.match(descriptor_1, descriptor_2, matches);

	//�滭
	Mat resImg;
	drawMatches(src, keyPoint_1, img_1, keyPoint_2, matches, resImg);
	imshow("res", resImg);

	*/

	#pragma endregion

	#pragma region FLANN����������
	/*
	
	//���ã�ƥ������ͼ�� src & img_1 �������res_img

	//��Ҫ����������
	//���β���SURF������
	Ptr<SURF> detector = SURF::create(400);
	//�������������ӵ�keypoint
	vector<KeyPoint> keyPoint_1;
	vector<KeyPoint> keyPoint_2;

	//��������������
	Mat descriptor_1, descriptor_2;
	detector->detectAndCompute(src, Mat(), keyPoint_1, descriptor_1);
	detector->detectAndCompute(img_1, Mat(), keyPoint_2, descriptor_2);

	//ƥ��
	FlannBasedMatcher flannMatches;
	vector<DMatch> matches;//ƥ���˷����
	flannMatches.match(descriptor_1, descriptor_2, matches);

	//�ҵ��õ�ƥ���filter good matches point ����
	//�����˸�BFMatchЧ��һ�������ǻ���
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
	}//������ǻ��������Сֵ����������е����һ��
	printf("max:%f\n", maxDist);
	printf("min:%f\n", minDist);

	vector<DMatch> goodMatch;
	for (int j = 0; j < descriptor_1.rows; j++)
	{
		double dist = matches[j].distance;
		if (dist < max(2 * minDist, 0.02))//����Ǹ���ģ�
		{
			goodMatch.push_back(matches[j]);
		}
	}

	Mat resImg;
	drawMatches(src, keyPoint_1, img_1, keyPoint_2, goodMatch, resImg);
	imshow("res img", resImg);

	

	*/
	#pragma endregion

	//findHomograph �� ��������ƽ���͸�ӱ任�����ɱ任����
	//perspectiveTransform ͸�ӱ任

	#pragma region AKAZE �ֲ�ƥ��

	/*

	//�ٶȸ��죬��SURF��SIFT�Ƚ�
	//AOS����߶ȿռ�
	//Hessian ������������
	//����ָ������һ��΢��ͼ��
	//����������
	//KAZE AKAZE

	//��ȡͼ��
	//Create����
	Ptr<KAZE> detector = KAZE::create();
	vector<KeyPoint> keyPoints;
	detector->detect(src, keyPoints);
	
	printf("KAZE");
	
	Mat keyPointImage;
	drawKeypoints(src, keyPoints, keyPointImage, Scalar::all(-1), DrawMatchesFlags::DEFAULT);;
	imshow("kaze res", keyPointImage);

	*/				
	#pragma endregion
	
	#pragma region FaceDetection �������
	/*

	//��Surface����Ҫ����λ�ã��õ����Դ���faceѵ������
	String caPath = "F:/OCV/opencv/newbuild/install/etc/haarcascades/haarcascade_frontalface_alt.xml";
	CascadeClassifier face_cascade;
	if (!face_cascade.load(caPath))
	{
		printf("face cascade could not load");
		waitKey(0);
		return -1;
		
	}

	cvtColor(src, graySrc, COLOR_BGR2GRAY);
	equalizeHist(graySrc, graySrc);

	vector<Rect> faces;
	face_cascade.detectMultiScale(graySrc, faces,1.1,2,0,Size(30,30));
	for (size_t i = 0; i < faces.size(); i++)
	{
		rectangle(src, faces[i], Scalar(0, 0, 255));
	}
	namedWindow("face detection", WINDOW_AUTOSIZE);
	imshow("face detection", src);

	*/
	#pragma endregion
	
	#pragma region ͼ����λ�б�

	
	Check_Skew();


	#pragma endregion

	
	
	
	waitKey(0);
	return 0;
}

void HarrisTrack(int, void *)
{
	dst = Mat::zeros(graySrc.size(), CV_32FC1);
	
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

void findROI(int, void *)//Ѱ�ұ�Ե
{
	cvtColor(src, graySrc, COLOR_BGR2GRAY);
	Mat canny_output;
	Canny(graySrc, canny_output, contours_TH, contours_TH * 2);
	vector<vector<Point>>contours;
	vector<Vec4i> hierachy;
	findContours(canny_output, contours, hierachy, RETR_TREE, CHAIN_APPROX_SIMPLE, Point(0));

}

void Check_Skew()
{
	/*
	Mat canny_output;
	Canny(graySrc, canny_output, skew_TH, skew_TH * 2);

	vector<vector<Point>> contours;
	vector<Vec4i> hierachy;
	findContours(canny_output, contours, hierachy, RETR_TREE, CHAIN_APPROX_SIMPLE, Point(0, 0));
	Mat drawImg = Mat::zeros(src.size(), CV_8UC3);

	float maxWidth = 0;
	float maxHeight = 0;
	double degree = 0;
	for (size_t i = 0; i < contours.size(); i++)
	{
		RotatedRect minRect = minAreaRect(contours[i]);
		degree = abs(minRect.angle);//�п����Ǹ�ֵ ��������abs����ֵ
		if (degree > 0)
		{
			maxWidth = max(maxWidth, minRect.size.width);
			maxHeight = max(maxHeight, minRect.size.width);

		}
		RNG rng(12345);//�����ɫ
		
		
		
	}
	*/



	Mat canny_output;
	cvtColor(src, graySrc, COLOR_BGR2GRAY);         //��ԭͼת��Ϊ�Ҷ�ͼ
	Canny(graySrc, canny_output, skew_TH, skew_TH * 2, 3, false);      // canny��Ե���
	imshow("ca", canny_output);
	vector<vector<Point>> contours;
	vector<Vec4i> hireachy;
	findContours(canny_output, contours, hireachy, RETR_TREE, CHAIN_APPROX_SIMPLE, Point(0, 0));    // �ҵ���������
	Mat drawImg = Mat::zeros(src.size(), CV_8UC3);
	float max_width = 0;       // ���������
	float max_height = 0;      // �������߶�
	double degree = 0;         // ������ת�Ƕ�
	for (auto t = 0; t < contours.size(); ++t)            // ����ÿһ������   
	{
		RotatedRect minRect = minAreaRect(contours[t]);        // �ҵ�ÿһ����������С�����ת���Σ�RotatedRect����������������ꡢ�ߴ��Լ���ת�Ƕȵ���Ϣ   
		degree = abs(minRect.angle);
		if (degree > 0)
		{
			max_width = max(max_width, minRect.size.width);
			max_height = max(max_height, minRect.size.height);
		}
	}
	RNG rng(12345);
	for (auto t = 0; t < contours.size(); ++t)
	{
		RotatedRect minRect = minAreaRect(contours[t]);
		if (max_width == minRect.size.width && max_height == minRect.size.height)
		{
			degree = minRect.angle;   // ����Ŀ�������ĽǶ�
			Point2f pts[4];
			minRect.points(pts);
			Scalar color = Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));  //���������ɫ
			for (int i = 0; i < 4; ++i)
			{
				line(drawImg, pts[i], pts[(i + 1) % 4], color, 2, 8, 0);
			}
		}
	}

	imshow("�ҵ��ľ�������", drawImg);
	//Point2f center(src.cols / 2, src.rows / 2);
	//Mat rotm = getRotationMatrix2D(center, degree, 1.0);    //��ȡ����任����
	//Mat dst;
	//warpAffine(src, dst, rotm, src.size(), INTER_LINEAR, 0, Scalar(255, 255, 255));    // ����ͼ����ת����
	//imwrite("C:/Users/SANG-Surface/Desktop/123.png", dst);      //��У�����ͼ�񱣴�����
	//imshow("Correct Image", dst);

	
}



