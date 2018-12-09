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
Mat src, dst, img_1,graySrc,normDst, gray_img;
Mat src_img = imread("./processPics/123.png");
//变量声明
int harrisThresh = 127;
int threshMax = 255;
//最大角点数
int maxCorner = 5;
int contours_TH = 127;
int skew_TH = 30;
int threshold_value = 100;
string output_win = "Contours Result";
string roi_win = "Final Result";

//方法声明
void HarrisTrack(int, void *);
void ShiTomasiTrack(int, void *);
void findROI(int, void *);//寻找边缘
void Check_Skew();



int main(int argc, char** argv) {
	//这里需要注意：在Surface上是SANG-Surface
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

	#pragma region FLANN过滤特征点
	/*
	
	//作用：匹配两张图像 src & img_1 最后生成res_img

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

	

	*/
	#pragma endregion

	//findHomograph ： 发现两个平面的透视变换，生成变换矩阵
	//perspectiveTransform 透视变换

	#pragma region AKAZE 局部匹配

	/*

	//速度更快，比SURF　SIFT比较
	//AOS构造尺度空间
	//Hessian 矩阵特征点检测
	//方向指定基于一阶微分图像
	//生成描述子
	//KAZE AKAZE

	//读取图像
	//Create对象
	Ptr<KAZE> detector = KAZE::create();
	vector<KeyPoint> keyPoints;
	detector->detect(src, keyPoints);
	
	printf("KAZE");
	
	Mat keyPointImage;
	drawKeypoints(src, keyPoints, keyPointImage, Scalar::all(-1), DrawMatchesFlags::DEFAULT);;
	imshow("kaze res", keyPointImage);

	*/				
	#pragma endregion
	
	#pragma region FaceDetection 人脸检测
	/*

	//在Surface上需要更改位置（用的是自带的face训练集）
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
	
	#pragma region 图像正位切边

	//https://blog.csdn.net/weixin_41695564/article/details/80077706
	
	//Check_Skew();

	/*namedWindow(output_win, WINDOW_AUTOSIZE);
	createTrackbar("Threshold:", output_win, &threshold_value, 255, findROI);
	findROI(0, 0);*/
	
	#pragma endregion

	#pragma region LineDetection 直线检测
	
	//Mat src_output;
	//src.convertTo(src_output, -1,3,0);//3是亮度
	//imshow("src_out2", src_output);//现在这个亮度还不错
	//
	////二值化
	//cvtColor(src_output, graySrc, CV_BGR2GRAY);
	//threshold(graySrc, graySrc, 0,255, THRESH_BINARY_INV | THRESH_OTSU);
	//imshow("th_out", graySrc);

	////形态学操作
	//Mat kernel = getStructuringElement(MORPH_RECT, Size(20, 1));
	//
	//morphologyEx(graySrc, graySrc, MORPH_OPEN, kernel);
	//imshow("mor", graySrc);

	////霍夫直线检测
	////注意api
	//vector<Vec4i> lines;
	//HoughLinesP(graySrc, lines,1,CV_PI / 180.0,10,7.0,0);
	//for (size_t i = 0; i < lines.size(); i++)
	//{
	//	Vec4i currPoints = lines[i];
	//	//这里可以骚着写
	//	line(src, Point(currPoints[0],currPoints[1]), Point(lines[i][2],lines[i][3]), Scalar(0, 0, 255));

	//}
	//imshow("src", src);

	#pragma endregion

	#pragma region ObjectsCount 对象计数

	//sfseed.jpg为少量瓜子 sfseeds.jpg数量较多
	//
	Mat src_output;
	src.convertTo(src_output, -1,3,0);//3是亮度
	imshow("src_out2", src_output);//现在这个亮度还不错


	#pragma endregion

	
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

//Why: Find ROI 找不到轮廓，如果去掉if判断 又会出现很多多余的框框
void fXindROI(int, void *)//寻找边缘
{
	
	cout << "**************当前阈值：" << skew_TH << "******************************\n" << endl;
	Mat src_img = imread("./processPics/magz.jpg");
	Mat mBlur;
	medianBlur(src_img, mBlur, 11);
	//cvtColor(src_img, graySrc, COLOR_BGR2GRAY);      //将原图转化为灰度图
	Mat canny_output;
	Canny(mBlur, canny_output, skew_TH, skew_TH * 2, 3, false);                // canny边缘检测
	imshow("canny_output", canny_output);
	vector<vector<Point>> contours;
	vector<Vec4i> hierachy;
	findContours(canny_output, contours, hierachy, RETR_TREE, CHAIN_APPROX_SIMPLE, Point(0, 0));    // 调用API，找到轮廓
	
	// 筛选contours中的轮廓，我们需要最大的那个轮廓
	float min_width = src_img.cols*0.5;          // 矩形的最小宽度	
	float min_height = src_img.rows*0.5;         // 矩形的最小高度
	RNG rng(12345);                            //定义一个随机数产生器，用来产生不同颜色的矩形框
	Mat drawImage = Mat::zeros(src_img.size(), CV_8UC3);
	Rect bbox;

	double degree = 0;

	for (auto t = 0; t < contours.size(); ++t)            // 遍历每一个轮廓   
	{
		RotatedRect minRect = minAreaRect(contours[t]);        // 找到每一个轮廓的最小外包旋转矩形，RotatedRect里面包含了中心坐标、尺寸以及旋转角度等信息   
		degree = abs(minRect.angle);
	
			cout << "Contours:" << contours.size() << "Degree:" << degree << endl;
			min_width = max(min_height, minRect.size.width);
			min_height = max(min_height, minRect.size.height);
		
	}


	for (auto t = 0; t < contours.size(); ++t)        // 遍历每一个轮廓
	{
		RotatedRect minRect = minAreaRect(contours[t]);        // 找到每一个轮廓的最小外包旋转矩形，RotatedRect里面包含了中心坐标、尺寸以及旋转角度等信息
		degree = abs(minRect.angle);                    // 最小外包旋转矩形的旋转角度
		if (minRect.size.width > min_width && minRect.size.height > min_height )//&& minRect.size.width < (src_img.cols - 5)   //筛选最小外包旋转矩形
		//if(degree > 0)
		//if (min_width == minRect.size.width && min_height == minRect.size.height)
		{

			printf("current angle : %f\n", degree);
			Mat vertices;       // 定义一个4行2列的单通道float类型的Mat，用来存储旋转矩形的四个顶点
			boxPoints(minRect, vertices);    // 计算旋转矩形的四个顶点坐标
			bbox = boundingRect(vertices);   //找到输入点集的最小外包直立矩形，返回Rect类型
			cout << "最小外包矩形：" << bbox << endl;
			Scalar color = Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));   //产生随机颜色
			for (int i = 0; i < 4; i++)             // 遍历每个旋转矩形的四个顶点坐标
			{
				// 在相邻的顶点之间绘制直线
				Mat p1 = vertices.row(i); // 使用成员函数row(i)和col(j)得到矩阵的第i行或者第j列，返回值仍然是一个单通道的Mat类型
				int j = (i + 1) % 4;
				Mat p2 = vertices.row(j);
				Point p1_point = Point(p1.at<float>(0, 0), p1.at<float>(0, 1)); //将Mat类型的顶点坐标转换为Point类型
				Point p2_point = Point(p2.at<float>(0, 0), p2.at<float>(0, 1));
				line(drawImage, p1_point, p2_point, color, 2, 8, 0);    // 根据得到的四个顶点，通过连接四个顶点，将最小旋转矩形绘制出来
			}
		}
	}
	imshow("drawImg", drawImage);
	if (bbox.width > 0 && bbox.height > 0)
	{
		Mat roiImg = src_img(bbox);        //从原图中截取兴趣区域
		namedWindow("resultPic", CV_WINDOW_AUTOSIZE);
		imshow("resultPic", roiImg);
	}

	return;


}

//矫正位置
void Check_Skew()
{
	Mat canny_output;
	Mat mBlur;
	
	//直接Canny的效果不好
	//先模糊看看
	//然后把界限最好放清楚一点 使用了中值模糊
	
	medianBlur(src, mBlur,11);
	
	//cvtColor(src, graySrc, COLOR_BGR2GRAY);         //将原图转化为灰度图
	imshow("mb", mBlur);
	
	//Why: 不知道为啥 不能使用灰度图
	Canny(mBlur, canny_output, skew_TH, skew_TH * 2, 3, false);      // canny边缘检测
	imshow("ca", canny_output);



	vector<vector<Point>> contours;
	vector<Vec4i> hireachy;
	findContours(canny_output, contours, hireachy, RETR_TREE, CHAIN_APPROX_SIMPLE, Point(0, 0));    // 找到所有轮廓
	Mat drawImg = Mat::zeros(src.size(), CV_8UC3);
	float max_width = 0;       // 定义最大宽度
	float max_height = 0;      // 定义最大高度
	double degree = 0;         // 定义旋转角度
	//这个for是为了找最大的框
	for (auto t = 0; t < contours.size(); ++t)            // 遍历每一个轮廓   
	{
		RotatedRect minRect = minAreaRect(contours[t]);        // 找到每一个轮廓的最小外包旋转矩形，RotatedRect里面包含了中心坐标、尺寸以及旋转角度等信息   
		degree = abs(minRect.angle);
		max_width = max(max_width, minRect.size.width);
		max_height = max(max_height, minRect.size.height);
	}
	RNG rng(12345);
	//这是为了画出最大的框
	for (auto t = 0; t < contours.size(); ++t)
	{
		RotatedRect minRect = minAreaRect(contours[t]);
		if (max_width == minRect.size.width || max_height == minRect.size.height)
		{
			degree = minRect.angle;   // 保存目标轮廓的角度
		
		
			Point2f pts[4];
			minRect.points(pts);
			Scalar color = Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));  //产生随机颜色
			for (int i = 0; i < 4; ++i)
			{
				line(drawImg, pts[i], pts[(i + 1) % 4], color, 2, 8, 0);
			}
			cout << "Size:" << minRect.size <<"Degree:"<<degree << endl;
		}
		
	}

	imshow("找到的矩形轮廓", drawImg);
	Point2f center(src.cols / 2, src.rows / 2);
	Mat rotm = getRotationMatrix2D(center, degree, 1.0);    //获取仿射变换矩阵
	Mat dst;
	warpAffine(src, dst, rotm, src.size(), INTER_LINEAR, BORDER_REPLICATE, Scalar(255, 255, 255));    // 进行图像旋转操作
	imwrite("./processPics/123.png", dst);      //将校正后的图像保存下来
	imshow("correct image", dst);

	
}


void findROI(int, void*)
{
	
	printf("**************当前阈值：%d******************************\n", threshold_value);
	//cvtColor(src_img, gray_img, COLOR_BGR2GRAY);      //将原图转化为灰度图
	Mat canny_output;
	Mat mBlur;
	medianBlur(src, mBlur, 11);
	Canny(mBlur, canny_output, threshold_value, threshold_value * 2, 3, false);                // canny边缘检测
	imshow("canny_output", canny_output);
	vector<vector<Point>> contours;
	vector<Vec4i> hireachy;
	findContours(canny_output, contours, hireachy, RETR_TREE, CHAIN_APPROX_SIMPLE, Point(0, 0));    // 调用API，找到轮廓

	// 筛选contours中的轮廓，我们需要最大的那个轮廓
	float max_width = 0;       // 定义最大宽度
	float max_height = 0;      // 定义最大高度
	double degree = 0;         // 定义旋转角度
	//这个for是为了找最大的框
	for (auto t = 0; t < contours.size(); ++t)            // 遍历每一个轮廓   
	{
		RotatedRect minRect = minAreaRect(contours[t]);        // 找到每一个轮廓的最小外包旋转矩形，RotatedRect里面包含了中心坐标、尺寸以及旋转角度等信息   
		degree = abs(minRect.angle);
		max_width = max(max_width, minRect.size.width);
		max_height = max(max_height, minRect.size.height);
	}

	RNG rng(12345);                            //定义一个随机数产生器，用来产生不同颜色的矩形框
	Mat drawImage = Mat::zeros(src_img.size(), CV_8UC3);
	Rect bbox;
	for (auto t = 0; t < contours.size(); ++t)        // 遍历每一个轮廓
	{
		RotatedRect minRect = minAreaRect(contours[t]);        // 找到每一个轮廓的最小外包旋转矩形，RotatedRect里面包含了中心坐标、尺寸以及旋转角度等信息
		
		if ((minRect.size.width == max_width || minRect.size.height == max_height) && minRect.size.width > 620)   //筛选最小外包旋转矩形
		{
			printf("current angle : %f\n", degree);
			Mat vertices;       // 定义一个4行2列的单通道float类型的Mat，用来存储旋转矩形的四个顶点
			boxPoints(minRect, vertices);    // 计算旋转矩形的四个顶点坐标
			bbox = boundingRect(vertices);   //找到输入点集的最小外包直立矩形，返回Rect类型
			cout << "最小外包矩形：" << bbox << endl;
			Scalar color = Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));   //产生随机颜色
			for (int i = 0; i < 4; i++)             // 遍历每个旋转矩形的四个顶点坐标
			{
				// 在相邻的顶点之间绘制直线
				Mat p1 = vertices.row(i); // 使用成员函数row(i)和col(j)得到矩阵的第i行或者第j列，返回值仍然是一个单通道的Mat类型
				int j = (i + 1) % 4;
				Mat p2 = vertices.row(j);
				Point p1_point = Point(p1.at<float>(0, 0), p1.at<float>(0, 1)); //将Mat类型的顶点坐标转换为Point类型
				Point p2_point = Point(p2.at<float>(0, 0), p2.at<float>(0, 1));
				line(src, p1_point, p2_point, color, 2, 8, 0);    // 根据得到的四个顶点，通过连接四个顶点，将最小旋转矩形绘制出来
			}
		}
	}
	imshow(output_win, src);

	if (bbox.width > 0 && bbox.height > 0)
	{
		Mat roiImg = src_img(bbox);        //从原图中截取兴趣区域
		namedWindow(roi_win, CV_WINDOW_AUTOSIZE);
		imshow(roi_win, roiImg);
	}

	return;
}

