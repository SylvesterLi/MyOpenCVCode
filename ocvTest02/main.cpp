#include <iostream>
#include <opencv2/opencv.hpp>
#include <math.h>

using namespace cv;
using namespace std;



int ele_size = 3;
int canny_threshold = 100;
int binary_threshold = 128;

Mat src, dst, gray_src, laplance_src, sharpImg;
Mat baseT;
//VideoCapture vc(0);
void Trackbar_CallBack(int, void*);
void AdjustThreshold(int, void*);
void binary_callback(int, void*);
void Contours_DrawPoly(int, void*);

int main(int argc, char** argv)
{
	//����ͼ�����
	
	

	//src = imread("C:/Users/SANG-ASUS/Desktop/cards.png");
	src = imread("cards.png");
	//baseT = imread("C:/Users/SANG-ASUS/Desktop/baseT.png");
	//src1 = imread("C:/Users/SANG-ASUS/Desktop/11.png");
	//src2 = imread("C:/Users/SANG-ASUS/Desktop/22.png");
	if (!src.data)
	{
		printf("no image");
	}
	//��ʾԭͼ
	namedWindow("src image", CV_WINDOW_AUTOSIZE);
	imshow("src image", src);



	#pragma region Mask Demo 02 ��ģ����1 

		/*

		//��ȡPoint(Row,Col)
		//const uchar* current = src.ptr<uchar>()

		int cols = (src.cols - 1)*src.channels();
		int offsetx = src.channels();
		int rows = src.rows;

		//printf("%d",offsetx); 3
		//��1��ʼ��0�Ƿ�����ĵ�һ������ͬ��lengthҲҪ-1
		dst = Mat::zeros(src.size(), src.type());
		for (int row = 1; row < rows - 1; row++)
		{
			//�õ���ǰptr ǰһ��ptr ��һ��ptr
			const uchar* current = src.ptr<uchar>(row);
			const uchar* next = src.ptr<uchar>(row);
			const uchar* previous = src.ptr<uchar>(row);

			uchar* output = dst.ptr<uchar>(row);
			for (int col = offsetx; col < cols - 1; col++)
			{
				output[col] =saturate_cast<uchar> ( current[col] * 5.5 - (previous[col] + next[col] + current[col - offsetx] + current[col + offsetx]));
			}

		}

		*/

	#pragma endregion

	#pragma region Mask Demo 02 ��ģ����2



		//double t = getTickFrequency();

		//Mat kernel = (Mat_<char>(3, 3) << 0, -1, 0, -1, 6, -1, 0, -1, 0);
		//filter2D(src, dst, src.depth(), kernel);

		//double timeCount = (getTickFrequency() - t) / getTickFrequency();
		//printf("%f", timeCount);
		//namedWindow("Test OpenCV setup", CV_WINDOW_AUTOSIZE);
		////��ʾ�ô���
		//imshow("Test OpenCV setup", dst);
		////�ȴ�������������¹رմ˴���
		//waitKey(0);

	#pragma endregion

	#pragma region Scalar Demo 03 ��ͼƬ��ɫ

		/*
		dst = Mat(src.size(), src.type());
		dst = Scalar(0, 25, 255);
		dst = src.clone();
		namedWindow("dst Image", CV_WINDOW_AUTOSIZE);
		imshow("dst Image",dst);
		*/

	#pragma endregion	

	#pragma region RevertColor Demo 04 ��ת�Ҷ�ͼ

		////תΪ�Ҷ�ͼ��
		//cvtColor(src, gray_src, CV_BGR2GRAY);

		////�õ�Width �� Height
		////��ʵҲ���Բ�Ҫ
		//int width = gray_src.rows;
		//int height = gray_src.cols;

		////��ɫ
		//for (int row = 0; row < height; row++)
		//{
		//	for (int col = 0; col < width; col++)
		//	{
		//		int gray = gray_src.at<uchar>(row, col);
		//		gray_src.at<uchar>(row, col) = 255 - gray;
		//	}
		//}
		//

		////��ʾͼ��
		//namedWindow("Revert Image", CV_WINDOW_AUTOSIZE);
		//imshow("Revert Image", gray_src);

	#pragma endregion

	#pragma region Revert3Channels Demo 05 ��תͼ��
			
	//����dstͼ��
	dst.create(src.size(), src.type());
	//ѭ����ȡsrc�ĵ������ص��BGR��ֵ
	/*
	for (int row = 0; row < src.rows; row++)
	{
		for (int col = 0; col < src.cols; col++)
		{
			int b = src.at<Vec3b>(row, col)[0];
			int g = src.at<Vec3b>(row, col)[1];
			int r = src.at<Vec3b>(row, col)[2];
			dst.at<Vec3b>(row, col)[0] = 255 - b;
			dst.at<Vec3b>(row, col)[1] = 255 - g;
			dst.at<Vec3b>(row, col)[2] = 255 - r;
		}
	}
	*/

	/*
	//��ɫ
	bitwise_not(src, dst);
	namedWindow("dst", CV_WINDOW_AUTOSIZE);
	imshow("dst", dst);
	*/

	#pragma endregion

	#pragma region Remix Image Demo 06 ���ͼ��
	/*
	addWeighted(src1, 0.4, src2, 0.6, 0.0, dst, -1);
	imshow("11", dst);
	add(src1, src2, dst, Mat());
	imshow("22", dst);
	multiply(src1, src2, dst, 1.0);
	namedWindow("Blend", CV_WINDOW_AUTOSIZE);
	imshow("Blend", dst);
	*/
	#pragma endregion

	#pragma region Contrast Demo 07 ��������

	/*
	//�Ƚ���dst
	dst = Mat::zeros(src.size(), src.type());
	//�趨alpha �� ����ֱ����߾���
	float alpha = 1.2;
	float beta = 10;

	//��srcתΪ���߾��ȵ�Vec3F
	Mat m1;
	src.convertTo(m1, CV_32F);

	//��ȡsrc���ص�bgr
	for (int row = 0; row < src.rows; row++)
	{
		for (int  col = 0; col < src.cols; col++)
		{
			//��ȡ��ͨ����BGRֵ
			float b= m1.at<Vec3f>(row, col)[0];
			float g = m1.at<Vec3f>(row, col)[1];
			float r = m1.at<Vec3f>(row, col)[2];

			//����dst������
			dst.at<Vec3b>(row, col)[0] = saturate_cast<uchar>(b*alpha + beta);
			dst.at<Vec3b>(row, col)[1] = saturate_cast<uchar>(g*alpha + beta);
			dst.at<Vec3b>(row, col)[2] = saturate_cast<uchar>(r*alpha + beta);
		}
	}
	imshow("dst", dst);

	*/


#pragma endregion

	#pragma region Draw LinesText Demo 8 ��ͼ

	//��ֱ��
	/*
	Point p1 = Point(20, 30);
	Point p2 = Point(50, 70);
	Scalar color = Scalar(0, 0, 255);
	line(src, p1, p2, color, 2, LINE_8);
	imshow("src", src);
	*/

	//������
	/*
	Rect rec = Rect(200, 100, 300, 300);
	Scalar color = Scalar(0, 255, 255);
	rectangle(src, rec, color, 2, LINE_4);
	*/

	//����Բ
	/*
	ellipse(src, Point(src.cols / 2, src.rows / 2),Size(src.cols/4,src.rows/8),90,50,180, Scalar(0, 255, 255),2, LINE_4);
	imshow("src", src);
	*/

	//��Բ
	//circle(src, Point(200, 200), 50, Scalar(0, 255, 255), 2, LINE_4);

	//д��
	//putText(src, "Hellow", Point(200, 200), CV_FONT_HERSHEY_DUPLEX,1.0, Scalar(255, 15, 255), 1, 8);
	
	//imshow("src", src);

	#pragma endregion

	#pragma region Random Draw Lines 9 �������
	/*
	RNG rng(12345);
	Point pt1, pt2;
	imshow("line", src);
	for (int i = 0; i < 9999999; i++)
	{
		pt1.x = rng.uniform(0, src.cols);
		pt1.y = rng.uniform(0, src.rows);
		pt2.x = rng.uniform(0, src.cols);
		pt2.y = rng.uniform(0, src.rows);

		Scalar color = Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
		line(src, pt1, pt2, color, 2, LINE_8);
		
		if (waitKey(50) > 0)
		{
			break;
		}

		imshow("line", src);
	}
	
	*/

	#pragma endregion

	#pragma region ģ�����˹ģ��
	
	
	
	/*
	blur(src, dst, Size(1, 155), Point(-1, -1));	
	GaussianBlur(src, dst, Size(11, 11), 5, 5);

	imshow("GaussianBlur", dst);
	*/

	#pragma endregion

	#pragma region ��ֵ�˲�

	//medianBlur(src, dst, 33);
	//bilateralFilter(src, dst, 15, 400,333);
	//imshow("src", dst);

	
	
	#pragma endregion

	//ģ��
	/*Mat element = getStructuringElement(MORPH_RECT, Size(15, 15));
	erode(src, dst, element);
	imshow("ddd", dst);
	//namedWindow("Output_Img", CV_WINDOW_AUTOSIZE);
	//createTrackbar("Change It!", "Output_Img", &ele_size, 23, Tracebar_CallBack);
	*/

	//��������ͷ
	/*
	while (1)
	{
		Mat frame;
		vc >> frame;
		imshow("vc", frame);
		
		cvtColor(frame, gray_src, COLOR_BGR2GRAY);
		//blur(gray_src, dst, Size(7,7));
		Tracebar_CallBack(0, 0);
		Canny(src, dst, 0, 30,3);
		//imshow("sss", dst);
		
		waitKey(30);
	}
	*/

	//��̬ѧ����
	/*
	Mat kernel = getStructuringElement(MORPH_RECT, Size(11, 11), Point(-1, -1));
	morphologyEx(src, dst, CV_MOP_BLACKHAT, kernel);

	imshow("ex", dst);
	*/
		
	//��ȡ���ߡ�����
	/*
	Mat gary_img;
	cvtColor(src, gary_img, CV_BGR2GRAY);
	
	Mat bin_img;
	adaptiveThreshold(~gary_img, bin_img, 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY, 15, 0);
	imshow("bing", bin_img);
	
	//��������ߣ�����vertical��
	Mat hline = getStructuringElement(MORPH_RECT, Size(bin_img.rows / 16, 1), Point(-1, -1));
	Mat vline = getStructuringElement(MORPH_RECT, Size(1, bin_img.rows / 16), Point(-1, -1));
	
	Mat d_img;
	//erode(bin_img, d_img, vline, Point(-1, -1));
	//dilate(d_img, dst, vline, Point(-1, -1));
	morphologyEx(bin_img, dst, CV_MOP_OPEN, hline, Point(-1, -1));
	imshow("dstimgn", dst);
	*/
	
	//pyr
	/*Mat up, down;
	pyrDown(src, down,Size(src.cols/2,src.rows/2));
	imshow("Down", down);
	pyrUp(src, up, Size(src.cols*2, src.rows*2));
	imshow("Up", up);*/
	
	//DOG
	/*
	Mat g1, g2, finTemp;
	GaussianBlur(src, g1, Size(3, 3),0);
	GaussianBlur(g1, g2, Size(3, 3), 0);
	subtract(g2, g1, finTemp, Mat());
	//��һ��
	
	
	normalize(finTemp, dst, 255, 64, NORM_MINMAX);
	imshow("subs", dst);
	*/

	//BinaryImage
	/*
	namedWindow("OUT", CV_WINDOW_AUTOSIZE);
	
	createTrackbar("Adjust:", "OUT", &ele_size, 255, Trackbar_CallBack);
	Trackbar_CallBack(0, 0);
	*/

	#pragma region Sobel����
	
	//����������src dst gray_src src1 src2
	
	//GaussianBlur(src, dst, Size(3, 3), 0, 0);
	//cvtColor(dst, gray_src, CV_BGR2GRAY);
	
	//��Ե�õ��˸���ļ�ǿ
	//Scharr(gray_src, src1, CV_16S, 1, 0);
	//Scharr(gray_src, src2, CV_16S, 0, 1);

	//Sobel(gray_src, src1, CV_16S, 1, 0, 3);
	//Sobel(gray_src, src2, CV_16S, 0, 1, 3);

	//convertScaleAbs(src1, src1);
	//convertScaleAbs(src2, src2);
	//imshow("x", src1);
	//imshow("y", src2);
	
	//ͼ����
	//addWeighted(src1, 0.5, src2, 0.5, 0, dst);

	//�Լ�дһ������㷨
	//����ʹ��dst����Ϊdst�ڴ�֮ǰ�Ѿ��涨����Mat��ʽ
	
	/*
	Mat picf = Mat(src1.size(), src1.type());
	for (int row = 0; row < src1.rows; row++)
	{
		for (int col = 0; col < src2.cols; col++)
		{
			int p1 = src1.at<uchar>(row, col);
			int p2 = src2.at<uchar>(row, col);
			picf.at<uchar>(row, col) = saturate_cast<uchar>(p1 + p2);
		}
	}
	*/
	
	//namedWindow("reWin", WINDOW_AUTOSIZE);
	//imshow("reWin", picf);
	
	#pragma endregion
	
	#pragma region Laplacian ������˹����

	/*
	GaussianBlur(src, dst, Size(3, 3), 0, 0);
	cvtColor(dst, gray_src, CV_BGR2GRAY);
	threshold(gray_src, gray_src,0, 255, THRESH_OTSU|THRESH_BINARY);
	Laplacian(gray_src, dst,CV_16S, 3);
	convertScaleAbs(dst, dst);


	imshow("output", dst);
	*/

	#pragma endregion
	
	#pragma region Canny ��Ե���

	/*
	Mat edge_mask;
	int va = 50;

	//cvtColor(src, gray_src,CV_BGR2GRAY);


	Canny(src, edge_mask, va *1, va*2, 3, false);
	dst.create(src.size(), src.type());
	src.copyTo(dst, edge_mask);
	imshow("o", dst);
	*/

	#pragma endregion
	
	#pragma region HoughLinsP ����ֱ�߱任
	/*

	//�Ƚ���Canny��Ե���
	Mat edge_src;
	vector<Vec4f> plines;

	Canny(src, edge_src, 100, 200);
	//����Ե����ֱ�߱任
	HoughLinesP(edge_src, plines, 1, CV_PI / 180.0, 5);
	//����
	for (size_t i = 0; i < plines.size(); i++)
	{
		Vec4f hlines = plines[i];
		line(src, Point(hlines[0], hlines[1]), Point(hlines[2], hlines[3]), Scalar(15,255,59), 10);
	}
	imshow("OUT", src);
	*/

	#pragma endregion

	#pragma region HoughCircle ����Բ���
	
	//����˵��
	/* --����CSDN--

	���ĸ�������double���͵�dp���������Բ�ĵ��ۼ���ͼ��ķֱ���������ͼ��֮�ȵĵ������Ҵ˲���������һ��������ͼ��ֱ��ʵ͵��ۼ������������ֲ������Ļ����������Ӱɡ����磬���dp= 1ʱ���ۼ���������ͼ�������ͬ�ķֱ��ʡ����dp=2���ۼ�����������ͼ��һ����ô��Ŀ�Ⱥ͸߶ȡ�

	�����������double���͵�minDist��Ϊ����任��⵽��Բ��Բ��֮�����С���룬�������ǵ��㷨���������ֵ�������ͬԲ֮�����С���롣����������̫С�Ļ���������ڵ�Բ���ܱ�����ؼ�����һ���غϵ�Բ����֮�������������̫��Ļ���ĳЩԲ�Ͳ��ܱ��������ˡ�

	������������double���͵�param1����Ĭ��ֵ100�����ǵ���������method���õļ�ⷽ���Ķ�Ӧ�Ĳ������Ե�ǰΨһ�ķ��������ݶȷ�CV_HOUGH_GRADIENT������ʾ���ݸ�canny��Ե������ӵĸ���ֵ��������ֵΪ����ֵ��һ�롣

	���߸�������double���͵�param2��Ҳ��Ĭ��ֵ100�����ǵ���������method���õļ�ⷽ���Ķ�Ӧ�Ĳ������Ե�ǰΨһ�ķ��������ݶȷ�CV_HOUGH_GRADIENT������ʾ�ڼ��׶�Բ�ĵ��ۼ�����ֵ����ԽС�Ļ����Ϳ��Լ�⵽������������ڵ�Բ������Խ��Ļ�����ͨ������Բ�͸��ӽӽ�������Բ���ˡ�

	�ڰ˸�������int���͵�minRadius,��Ĭ��ֵ0����ʾԲ�뾶����Сֵ��

	�ھŸ�������int���͵�maxRadius,Ҳ��Ĭ��ֵ0����ʾԲ�뾶�����ֵ����Ҫע����ǣ�ʹ�ô˺������Ժ����׵ؼ���Բ��Բ�ģ������������Ҳ������ʵ�Բ�뾶
	*/
	
	/*

	//�Ƚ����˲� ���ѡ����ֵ�˲�
	medianBlur(src, dst, 3);
	cvtColor(dst,gray_src, CV_BGR2GRAY);
	//����Բ���
	vector<Vec3f> pCircle;
	//              1         2            3          4  5   6    7 8  9
	HoughCircles(gray_src, pCircle, CV_HOUGH_GRADIENT,1, 5, 100, 65,80,100);
	

	
	cvtColor(gray_src, gray_src, CV_GRAY2BGR);
	for (size_t i = 0; i < pCircle.size(); i++)
	{
		Vec3f hlines = pCircle[i];
		circle(gray_src, Point(hlines[0], hlines[1]), hlines[2], Scalar(15, 255, 59), 2, LINE_AA);

	}

	imshow("OUT", gray_src);

	*/

	#pragma endregion

	#pragma region  HistogramCalculate ֱ��ͼ����

	/*

	//����ͷ�ۣ�ûɶů��
	//��ͨ����ʾ
	vector<Mat> bgrSrc;
	split(src, bgrSrc);

	//calcHist
	calcHist(bgrSrc,)

	*/

	#pragma endregion
	
	#pragma region TemplateMatch ģ��ƥ��
	
	//������Ĵ����װ��MatchingMethod�Ϳ�����
	// createTrackbar( trackbar_label, image_window, &match_method, max_Trackbar, MatchingMethod );
	//MatchingMethod(0, 0);

	/*
	double minVal, maxVal;
	//ģ��ƥ�����SQDIFF��SQDIFF_NORMED��ԽС����ֵ���Ÿ��ߵ�ƥ��Ч���������ķ�����ֵԽ���ƥ��Ч��Խ��
	matchTemplate(src, baseT, dst, CV_TM_SQDIFF_NORMED);
	normalize(dst, dst, 0, 1, NORM_MINMAX);

	Point minLoc, maxLoc, matchLoc;
	//minMaxLoc��ͼ�����ҵ����ֵ����Сֵ�����Ҵ����minLoc��maxLoc��
	minMaxLoc(dst, &minVal, &maxVal, &minLoc, &maxLoc, Mat());
	rectangle(src, minLoc, Point(minLoc.x + baseT.cols, minLoc.y + baseT.rows), Scalar(0, 255, 0));
	cout << "ƥ��ȣ�" << minVal << endl;
	imshow("wht", src);
	*/


	#pragma endregion

	#pragma region FindContours ��������
	/*
	cvtColor(src, gray_src, CV_BGR2GRAY);
	namedWindow("out", WINDOW_AUTOSIZE);
	//��ʱ���лҶ�ͼ��canny_thresholdĬ��Ϊ3
	createTrackbar("track", "out", &canny_threshold, 500, AdjustThreshold);
	//��Ҫע����Ҫ��ǰ������
	AdjustThreshold(0, 0);
	imshow("out", dst);
	*/
	#pragma endregion

	#pragma region ConvexHull ͹��

	/*
	namedWindow("out", WINDOW_AUTOSIZE);
	cvtColor(src, gray_src, CV_BGR2GRAY);
	blur(gray_src, gray_src, Size(3, 3));
	createTrackbar("binary","out", &binary_threshold, 255, binary_callback);
	binary_callback(0, 0);
	imshow("out", dst);
	*/

	#pragma endregion

	#pragma region DrawPoly��������λ���

	/*
	namedWindow("out", WINDOW_AUTOSIZE);
	cvtColor(src, gray_src, CV_BGR2GRAY);
	blur(gray_src, gray_src, Size(3, 3));
	createTrackbar("binary", "out", &binary_threshold, 255, Contours_DrawPoly);
	Contours_DrawPoly(0, 0);
	imshow("out", src);
	*/
	#pragma endregion

	#pragma region ͼ��ָ�

		//namedWindow("dst", WINDOW_AUTOSIZE);

		#pragma region ��ɺ�ɫ����
		/*
		for (size_t i = 0; i < src.rows; i++)
		{
			for (size_t j = 0; j < src.cols; j++)
			{
				if (src.at<Vec3b>(i, j) == Vec3b(255, 255, 255))
				{
					src.at<Vec3b>(i, j) = Vec3b(0, 0, 0);
				}
			}
		}
		//��ʱ���Ǻ�ɫ����
		//imshow("black image", src);
		#pragma endregion	

		#pragma region ��

		//����һ��������˹����
		Mat kernel = (Mat_<float>(3, 3) << 1, 1, 1, 1, -8, 1, 1, 1, 1);
		filter2D(src, laplance_src, CV_32F, kernel);
		//imshow("lap", laplance_src);


		//32F��ͼ�����sharpImg
		src.convertTo(sharpImg, CV_32F);

		//����Ҫ�������ɫ������ͼ��black src����Ȼ�����������
		sharpImg = sharpImg - laplance_src;
		sharpImg.convertTo(sharpImg, CV_8UC3);
		//imshow("AfterSharp", sharpImg);

		*/
		#pragma endregion

		#pragma region ��ֵ�� ����仯 ��һ��
		/*
		//���ڵõ���sharpImg��8UC3��ʽ src��Ȼ��black src dst��ʱ��û���� Laplance_src��������ò�����
		//׼�����е����� ��ֵ��
		//��תΪ��ɫͼ��
		cvtColor(sharpImg, sharpImg, CV_BGR2GRAY);
		//���ж�ֵ������ʵ40Ч�������ã��Ҿ���140���ԣ���������˸�THRESH_OTSU�����Ҳ��һ��������Ҫע��һ�£�
		threshold(sharpImg, gray_src, 40, 255, THRESH_BINARY|THRESH_OTSU);
		//���о���任
		distanceTransform(gray_src, dst, DIST_L2, 3);
		//�任���˻���������������ǹ�һ��֮�����Ч����
		normalize(dst, dst, 0, 1, NORM_MINMAX);
		//��ֵ���õ���ű�ǣ�ȷʵ�Ǻܳ���ı���ˣ�
		threshold(dst, dst, 0.4, 1, THRESH_BINARY);
		//��ʴ
		Mat kernel1 = Mat::ones(3, 3, CV_8UC1);
		erode(dst, dst, kernel1);


		
		dst.convertTo(dst,CV_8U);
		Mat marks = Mat::zeros(dst.size(), CV_32SC1);
		//��������

		//���ڱ����ҵ�������
		vector<vector<Point>> contours;
		findContours(dst, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
		//��һ����������� ��������������ɫ  �����ɫ�������ͬ����
		
		for (size_t i = 0; i < contours.size(); i++)
		{
			Scalar color = Scalar::all(static_cast<int>(i)+1);
			drawContours(marks, contours, i, color, -1);//-1Ϊ����ڲ�
		}
		circle(marks, Point(5, 5), 3, Scalar(255, 255, 255), -1);
		//չʾ����
		imshow("mkr",marks* 1000);

		
		//��ʼ��ˮ��任
		watershed(src, marks);
		//��*1000����ȫ�� *���Ժ󱳾����ǰ�ɫ�ģ�cards����ʾ������
		imshow("water", marks * 1000);
		//���漸�д���ûɶ�ã����Ǹ��㿴����ˮ���Ч��
		Mat msk = Mat::zeros(marks.size(), CV_8UC1);
		marks.convertTo(msk, CV_8UC1);
		bitwise_not(msk, msk,Mat());
		imshow("mak", msk);

		
		//��ʼ��ɫ �õ��Ƿ�ˮ����marks
		vector<Vec3b> colors;
		//׼��һ�������ɫ
		for (size_t i = 0; i < contours.size(); i++)
		{
			int r = theRNG().uniform(0, 255);
			int g = theRNG().uniform(0, 255);
			int b = theRNG().uniform(0, 255);
			colors.push_back(Vec3b((uchar)b, (uchar)g, (uchar)r));

		}
		
		dst = Mat::zeros(marks.size(), CV_8UC3);
		
		for (int i = 0; i < marks.rows; i++)
		{
			for (int j = 0; j < marks.cols; j++)
			{
				int index = marks.at<int>(i, j);

				if (index>0 && index<=static_cast<int>(contours.size()))
				{
					dst.at<Vec3b>(i, j) = colors[index - 1];
				}
				else
				{
					dst.at<Vec3b>(i, j) = Vec3b(0, 0, 0);
				}
			}
		}
		*/
		
		imshow("dst", dst);
		#pragma endregion

	#pragma endregion


	waitKey(0);


	return 0;
}

void Trackbar_CallBack(int,void*)
{
	int s = ele_size;
	cvtColor(src, gray_src, CV_BGR2GRAY);
	//�Զ�Ѱ�Һ��ʵ���ֵTHRESH_TRIANGLE|THRESH_BINARY
	threshold(gray_src, dst, s, 255,THRESH_BINARY);
	imshow("OUT", dst);
	return;
}

void Tracebar_CallBack(int, void*)
{
	int s = ele_size*2+1;
	Mat element = getStructuringElement(MORPH_RECT, Size(s, s), Point(-1, -1));
	dilate(gray_src, src, element, Point(-1, -1), 1);
	imshow("Output_Img", dst);
	return;
}

//��������Callback
void AdjustThreshold(int, void*)
{
	//canny_threshold
	Mat canout;
	Canny(gray_src, canout, canny_threshold, canny_threshold * 2);
	//vector֪ʶ��� https://blog.csdn.net/u010368556/article/details/79179669
	//hierachy �ȼ��ƶ�
	//���ڱ����ҵ���ͼ��ȼ�
	vector<Vec4i> hierachy;
	//���ڱ����ҵ�������
	vector<vector<Point>> contours;
	
	//֪ʶ��� https://blog.csdn.net/keith_bb/article/details/70185209
	//��������ģʽ
	/*
	RETR_EXTERNAL:��ʾֻ����������������������������hierarchy[i][2]=hierarchy[i][3]=-1 
	RETR_LIST:��ȡ������������������list�У����������������ȼ���ϵ 
	RETR_CCOMP:��ȡ��������������������֯��˫��ṹ(two-level hierarchy),����Ϊ��ͨ�����Χ�߽磬�β�λ�ڲ�߽� 
	RETR_TREE:��ȡ�������������½�����״�����ṹ 
	RETR_FLOODFILL������û�н��ܣ�Ӧ���Ǻ�ˮ��䷨ 
	*/	
	//�������Ʒ���
	/*
	CHAIN_APPROX_NONE����ȡÿ��������ÿ�����أ����ڵ������������λ�ò����1 
	CHAIN_APPROX_SIMPLE��ѹ��ˮƽ���򣬴�ֱ���򣬶Խ��߷����Ԫ�أ�ֵ�����÷�����ص����꣬���һ����������ֻ��4����������������Ϣ 
	CHAIN_APPROX_TC89_L1��CHAIN_APPROX_TC89_KCOSʹ��Teh-Chinl���ƽ��㷨�е�һ��
	*/
	findContours(canout, contours, hierachy, RETR_TREE, CHAIN_APPROX_SIMPLE);
	//��һ����������� ��������������ɫ  �����ɫ�������ͬ����
	RNG rng(12345);
	for (size_t i = 0; i < contours.size(); i++)
	{
		Scalar color = Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
		drawContours(dst, contours, i, color, 2, LINE_AA, hierachy);
	}
	imshow("out", dst);
}

//͹������
void binary_callback(int, void*)
{
	Mat canout;


	//���ڱ����ҵ���ͼ��ȼ�
	vector<Vec4i> hierachy;
	//���ڱ����ҵ�������
	vector<vector<Point>> contours;
	threshold(gray_src, canout, binary_threshold, 255, THRESH_BINARY);
	//��������
	findContours(canout, contours, hierachy, RETR_TREE, CHAIN_APPROX_SIMPLE);
	vector<vector<Point>> hulls(contours.size());
	for (size_t i = 0; i < contours.size(); i++)
	{
		convexHull(contours[i], hulls[i]);
	}
	dst = Mat::zeros(src.size(), CV_8UC3);

	RNG rng(12345);
	for (size_t i = 0; i < contours.size(); i++)
	{
		Scalar color = Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
		//drawContours(dst, contours, i, color, 2, LINE_AA, hierachy);
		drawContours(dst, hulls, i, color, 2, LINE_AA, hierachy);
	}
	imshow("out", dst);




}

//�������������
void Contours_DrawPoly(int, void*)
{
	Mat canout;
	//���ڱ����ҵ���ͼ��ȼ�
	vector<Vec4i> hierachy;
	//���ڱ����ҵ�������
	vector<vector<Point>> contours;
	
	threshold(gray_src, canout, binary_threshold, 255, THRESH_BINARY);
	//��������
	findContours(canout, contours, hierachy, RETR_TREE, CHAIN_APPROX_SIMPLE);
	
	


	//��ž���
	vector<Rect> rectangles(contours.size());
	//���
	vector<Point2f> centers(contours.size());
	vector<float> radiux(contours.size());

	//��С����
	vector<vector<Point>> contours_covs(contours.size());
	for (size_t i = 0; i < contours.size(); i++)
	{
		approxPolyDP(contours[i], contours_covs[i], 3, true);
		//��һ�е��������һ�����Σ�������ת����һ���㣩
		rectangles[i] = boundingRect(contours_covs[i]);
		//��contours��ĵ������Բ�����Բ��������뾶
		minEnclosingCircle(contours_covs[i], centers[i], radiux[i]);

	}
	//����
	RNG rng(12345);
	for (size_t j = 0; j < contours_covs.size(); j++)
	{
		Scalar color = Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
		rectangle(src, rectangles[j], color);
		circle(src, centers[j], radiux[j], color);

	}

	

	/*
	vector<vector<Point>> contours_ploy(contours.size());
	vector<Rect> ploy_rects(contours.size());
	vector<Point2f> ccs(contours.size());
	vector<float> radius(contours.size());

	vector<RotatedRect> minRects(contours.size());
	vector<RotatedRect> myellipse(contours.size());

	for (size_t i = 0; i < contours.size(); i++) {
		approxPolyDP(Mat(contours[i]), contours_ploy[i], 3, true);
		ploy_rects[i] = boundingRect(contours_ploy[i]);
		minEnclosingCircle(contours_ploy[i], ccs[i], radius[i]);
		if (contours_ploy[i].size() > 5) {
			myellipse[i] = fitEllipse(contours_ploy[i]);
			minRects[i] = minAreaRect(contours_ploy[i]);
		}
	}

	// draw it
	//drawImg = Mat::zeros(src.size(), src.type());
	RNG rng(12345);
	Point2f pts[4];
	for (size_t t = 0; t < contours.size(); t++) {
		Scalar color = Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
		rectangle(src, ploy_rects[t], color, 2, 8);
		circle(src, ccs[t], radius[t], color, 2, 8);
		if (contours_ploy[t].size() > 5) {
			ellipse(src, myellipse[t], color, 1, 8);
			minRects[t].points(pts);
			for (int r = 0; r < 4; r++) {
				line(src, pts[r], pts[(r + 1) % 4], color, 1, 8);
			}
		}
	}
	*/

	
	imshow("out", src);

}