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
Mat src, dst, img_1,graySrc,normDst, gray_img;
Mat src_img = imread("./processPics/123.png");
//��������
int harrisThresh = 127;
int threshMax = 255;
//���ǵ���
int maxCorner = 5;
int contours_TH = 127;
int skew_TH = 30;
int threshold_value = 100;
string output_win = "Contours Result";
string roi_win = "Final Result";

//��������
void HarrisTrack(int, void *);
void ShiTomasiTrack(int, void *);
void findROI(int, void *);//Ѱ�ұ�Ե
void Check_Skew();



int main(int argc, char** argv) {
	//������Ҫע�⣺��Surface����SANG-Surface
	src = imread("./processPics/sfseed.jpg");
	//img_1 = imread("./processPics/magz.jpg");
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

	//https://blog.csdn.net/weixin_41695564/article/details/80077706
	
	//Check_Skew();

	/*namedWindow(output_win, WINDOW_AUTOSIZE);
	createTrackbar("Threshold:", output_win, &threshold_value, 255, findROI);
	findROI(0, 0);*/
	
	#pragma endregion

	#pragma region LineDetection ֱ�߼��
	
	//Mat src_output;
	//src.convertTo(src_output, -1,3,0);//3������
	//imshow("src_out2", src_output);//����������Ȼ�����
	//
	////��ֵ��
	//cvtColor(src_output, graySrc, CV_BGR2GRAY);
	//threshold(graySrc, graySrc, 0,255, THRESH_BINARY_INV | THRESH_OTSU);
	//imshow("th_out", graySrc);

	////��̬ѧ����
	//Mat kernel = getStructuringElement(MORPH_RECT, Size(20, 1));
	//
	//morphologyEx(graySrc, graySrc, MORPH_OPEN, kernel);
	//imshow("mor", graySrc);

	////����ֱ�߼��
	////ע��api
	//vector<Vec4i> lines;
	//HoughLinesP(graySrc, lines,1,CV_PI / 180.0,10,7.0,0);
	//for (size_t i = 0; i < lines.size(); i++)
	//{
	//	Vec4i currPoints = lines[i];
	//	//�������ɧ��д
	//	line(src, Point(currPoints[0],currPoints[1]), Point(lines[i][2],lines[i][3]), Scalar(0, 0, 255));

	//}
	//imshow("src", src);

	#pragma endregion

	#pragma region ObjectsCount �������

	//sfseed.jpgΪ�������� sfseeds.jpg�����϶�
	//
	Mat src_output;
	src.convertTo(src_output, -1,3,0);//3������
	imshow("src_out2", src_output);//����������Ȼ�����


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

//Why: Find ROI �Ҳ������������ȥ��if�ж� �ֻ���ֺܶ����Ŀ��
void fXindROI(int, void *)//Ѱ�ұ�Ե
{
	
	cout << "**************��ǰ��ֵ��" << skew_TH << "******************************\n" << endl;
	Mat src_img = imread("./processPics/magz.jpg");
	Mat mBlur;
	medianBlur(src_img, mBlur, 11);
	//cvtColor(src_img, graySrc, COLOR_BGR2GRAY);      //��ԭͼת��Ϊ�Ҷ�ͼ
	Mat canny_output;
	Canny(mBlur, canny_output, skew_TH, skew_TH * 2, 3, false);                // canny��Ե���
	imshow("canny_output", canny_output);
	vector<vector<Point>> contours;
	vector<Vec4i> hierachy;
	findContours(canny_output, contours, hierachy, RETR_TREE, CHAIN_APPROX_SIMPLE, Point(0, 0));    // ����API���ҵ�����
	
	// ɸѡcontours�е�������������Ҫ�����Ǹ�����
	float min_width = src_img.cols*0.5;          // ���ε���С���	
	float min_height = src_img.rows*0.5;         // ���ε���С�߶�
	RNG rng(12345);                            //����һ�������������������������ͬ��ɫ�ľ��ο�
	Mat drawImage = Mat::zeros(src_img.size(), CV_8UC3);
	Rect bbox;

	double degree = 0;

	for (auto t = 0; t < contours.size(); ++t)            // ����ÿһ������   
	{
		RotatedRect minRect = minAreaRect(contours[t]);        // �ҵ�ÿһ����������С�����ת���Σ�RotatedRect����������������ꡢ�ߴ��Լ���ת�Ƕȵ���Ϣ   
		degree = abs(minRect.angle);
	
			cout << "Contours:" << contours.size() << "Degree:" << degree << endl;
			min_width = max(min_height, minRect.size.width);
			min_height = max(min_height, minRect.size.height);
		
	}


	for (auto t = 0; t < contours.size(); ++t)        // ����ÿһ������
	{
		RotatedRect minRect = minAreaRect(contours[t]);        // �ҵ�ÿһ����������С�����ת���Σ�RotatedRect����������������ꡢ�ߴ��Լ���ת�Ƕȵ���Ϣ
		degree = abs(minRect.angle);                    // ��С�����ת���ε���ת�Ƕ�
		if (minRect.size.width > min_width && minRect.size.height > min_height )//&& minRect.size.width < (src_img.cols - 5)   //ɸѡ��С�����ת����
		//if(degree > 0)
		//if (min_width == minRect.size.width && min_height == minRect.size.height)
		{

			printf("current angle : %f\n", degree);
			Mat vertices;       // ����һ��4��2�еĵ�ͨ��float���͵�Mat�������洢��ת���ε��ĸ�����
			boxPoints(minRect, vertices);    // ������ת���ε��ĸ���������
			bbox = boundingRect(vertices);   //�ҵ�����㼯����С���ֱ�����Σ�����Rect����
			cout << "��С������Σ�" << bbox << endl;
			Scalar color = Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));   //���������ɫ
			for (int i = 0; i < 4; i++)             // ����ÿ����ת���ε��ĸ���������
			{
				// �����ڵĶ���֮�����ֱ��
				Mat p1 = vertices.row(i); // ʹ�ó�Ա����row(i)��col(j)�õ�����ĵ�i�л��ߵ�j�У�����ֵ��Ȼ��һ����ͨ����Mat����
				int j = (i + 1) % 4;
				Mat p2 = vertices.row(j);
				Point p1_point = Point(p1.at<float>(0, 0), p1.at<float>(0, 1)); //��Mat���͵Ķ�������ת��ΪPoint����
				Point p2_point = Point(p2.at<float>(0, 0), p2.at<float>(0, 1));
				line(drawImage, p1_point, p2_point, color, 2, 8, 0);    // ���ݵõ����ĸ����㣬ͨ�������ĸ����㣬����С��ת���λ��Ƴ���
			}
		}
	}
	imshow("drawImg", drawImage);
	if (bbox.width > 0 && bbox.height > 0)
	{
		Mat roiImg = src_img(bbox);        //��ԭͼ�н�ȡ��Ȥ����
		namedWindow("resultPic", CV_WINDOW_AUTOSIZE);
		imshow("resultPic", roiImg);
	}

	return;


}

//����λ��
void Check_Skew()
{
	Mat canny_output;
	Mat mBlur;
	
	//ֱ��Canny��Ч������
	//��ģ������
	//Ȼ��ѽ�����÷����һ�� ʹ������ֵģ��
	
	medianBlur(src, mBlur,11);
	
	//cvtColor(src, graySrc, COLOR_BGR2GRAY);         //��ԭͼת��Ϊ�Ҷ�ͼ
	imshow("mb", mBlur);
	
	//Why: ��֪��Ϊɶ ����ʹ�ûҶ�ͼ
	Canny(mBlur, canny_output, skew_TH, skew_TH * 2, 3, false);      // canny��Ե���
	imshow("ca", canny_output);



	vector<vector<Point>> contours;
	vector<Vec4i> hireachy;
	findContours(canny_output, contours, hireachy, RETR_TREE, CHAIN_APPROX_SIMPLE, Point(0, 0));    // �ҵ���������
	Mat drawImg = Mat::zeros(src.size(), CV_8UC3);
	float max_width = 0;       // ���������
	float max_height = 0;      // �������߶�
	double degree = 0;         // ������ת�Ƕ�
	//���for��Ϊ�������Ŀ�
	for (auto t = 0; t < contours.size(); ++t)            // ����ÿһ������   
	{
		RotatedRect minRect = minAreaRect(contours[t]);        // �ҵ�ÿһ����������С�����ת���Σ�RotatedRect����������������ꡢ�ߴ��Լ���ת�Ƕȵ���Ϣ   
		degree = abs(minRect.angle);
		max_width = max(max_width, minRect.size.width);
		max_height = max(max_height, minRect.size.height);
	}
	RNG rng(12345);
	//����Ϊ�˻������Ŀ�
	for (auto t = 0; t < contours.size(); ++t)
	{
		RotatedRect minRect = minAreaRect(contours[t]);
		if (max_width == minRect.size.width || max_height == minRect.size.height)
		{
			degree = minRect.angle;   // ����Ŀ�������ĽǶ�
		
		
			Point2f pts[4];
			minRect.points(pts);
			Scalar color = Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));  //���������ɫ
			for (int i = 0; i < 4; ++i)
			{
				line(drawImg, pts[i], pts[(i + 1) % 4], color, 2, 8, 0);
			}
			cout << "Size:" << minRect.size <<"Degree:"<<degree << endl;
		}
		
	}

	imshow("�ҵ��ľ�������", drawImg);
	Point2f center(src.cols / 2, src.rows / 2);
	Mat rotm = getRotationMatrix2D(center, degree, 1.0);    //��ȡ����任����
	Mat dst;
	warpAffine(src, dst, rotm, src.size(), INTER_LINEAR, BORDER_REPLICATE, Scalar(255, 255, 255));    // ����ͼ����ת����
	imwrite("./processPics/123.png", dst);      //��У�����ͼ�񱣴�����
	imshow("correct image", dst);

	
}


void findROI(int, void*)
{
	
	printf("**************��ǰ��ֵ��%d******************************\n", threshold_value);
	//cvtColor(src_img, gray_img, COLOR_BGR2GRAY);      //��ԭͼת��Ϊ�Ҷ�ͼ
	Mat canny_output;
	Mat mBlur;
	medianBlur(src, mBlur, 11);
	Canny(mBlur, canny_output, threshold_value, threshold_value * 2, 3, false);                // canny��Ե���
	imshow("canny_output", canny_output);
	vector<vector<Point>> contours;
	vector<Vec4i> hireachy;
	findContours(canny_output, contours, hireachy, RETR_TREE, CHAIN_APPROX_SIMPLE, Point(0, 0));    // ����API���ҵ�����

	// ɸѡcontours�е�������������Ҫ�����Ǹ�����
	float max_width = 0;       // ���������
	float max_height = 0;      // �������߶�
	double degree = 0;         // ������ת�Ƕ�
	//���for��Ϊ�������Ŀ�
	for (auto t = 0; t < contours.size(); ++t)            // ����ÿһ������   
	{
		RotatedRect minRect = minAreaRect(contours[t]);        // �ҵ�ÿһ����������С�����ת���Σ�RotatedRect����������������ꡢ�ߴ��Լ���ת�Ƕȵ���Ϣ   
		degree = abs(minRect.angle);
		max_width = max(max_width, minRect.size.width);
		max_height = max(max_height, minRect.size.height);
	}

	RNG rng(12345);                            //����һ�������������������������ͬ��ɫ�ľ��ο�
	Mat drawImage = Mat::zeros(src_img.size(), CV_8UC3);
	Rect bbox;
	for (auto t = 0; t < contours.size(); ++t)        // ����ÿһ������
	{
		RotatedRect minRect = minAreaRect(contours[t]);        // �ҵ�ÿһ����������С�����ת���Σ�RotatedRect����������������ꡢ�ߴ��Լ���ת�Ƕȵ���Ϣ
		
		if ((minRect.size.width == max_width || minRect.size.height == max_height) && minRect.size.width > 620)   //ɸѡ��С�����ת����
		{
			printf("current angle : %f\n", degree);
			Mat vertices;       // ����һ��4��2�еĵ�ͨ��float���͵�Mat�������洢��ת���ε��ĸ�����
			boxPoints(minRect, vertices);    // ������ת���ε��ĸ���������
			bbox = boundingRect(vertices);   //�ҵ�����㼯����С���ֱ�����Σ�����Rect����
			cout << "��С������Σ�" << bbox << endl;
			Scalar color = Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));   //���������ɫ
			for (int i = 0; i < 4; i++)             // ����ÿ����ת���ε��ĸ���������
			{
				// �����ڵĶ���֮�����ֱ��
				Mat p1 = vertices.row(i); // ʹ�ó�Ա����row(i)��col(j)�õ�����ĵ�i�л��ߵ�j�У�����ֵ��Ȼ��һ����ͨ����Mat����
				int j = (i + 1) % 4;
				Mat p2 = vertices.row(j);
				Point p1_point = Point(p1.at<float>(0, 0), p1.at<float>(0, 1)); //��Mat���͵Ķ�������ת��ΪPoint����
				Point p2_point = Point(p2.at<float>(0, 0), p2.at<float>(0, 1));
				line(src, p1_point, p2_point, color, 2, 8, 0);    // ���ݵõ����ĸ����㣬ͨ�������ĸ����㣬����С��ת���λ��Ƴ���
			}
		}
	}
	imshow(output_win, src);

	if (bbox.width > 0 && bbox.height > 0)
	{
		Mat roiImg = src_img(bbox);        //��ԭͼ�н�ȡ��Ȥ����
		namedWindow(roi_win, CV_WINDOW_AUTOSIZE);
		imshow(roi_win, roiImg);
	}

	return;
}

