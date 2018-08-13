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
	//声明图像变量
	
	

	//src = imread("C:/Users/SANG-ASUS/Desktop/cards.png");
	src = imread("cards.png");
	//baseT = imread("C:/Users/SANG-ASUS/Desktop/baseT.png");
	//src1 = imread("C:/Users/SANG-ASUS/Desktop/11.png");
	//src2 = imread("C:/Users/SANG-ASUS/Desktop/22.png");
	if (!src.data)
	{
		printf("no image");
	}
	//显示原图
	namedWindow("src image", CV_WINDOW_AUTOSIZE);
	imshow("src image", src);



	#pragma region Mask Demo 02 掩模操作1 

		/*

		//获取Point(Row,Col)
		//const uchar* current = src.ptr<uchar>()

		int cols = (src.cols - 1)*src.channels();
		int offsetx = src.channels();
		int rows = src.rows;

		//printf("%d",offsetx); 3
		//从1开始（0是方格里的第一个），同理length也要-1
		dst = Mat::zeros(src.size(), src.type());
		for (int row = 1; row < rows - 1; row++)
		{
			//拿到当前ptr 前一个ptr 后一个ptr
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

	#pragma region Mask Demo 02 掩模操作2



		//double t = getTickFrequency();

		//Mat kernel = (Mat_<char>(3, 3) << 0, -1, 0, -1, 6, -1, 0, -1, 0);
		//filter2D(src, dst, src.depth(), kernel);

		//double timeCount = (getTickFrequency() - t) / getTickFrequency();
		//printf("%f", timeCount);
		//namedWindow("Test OpenCV setup", CV_WINDOW_AUTOSIZE);
		////显示该窗口
		//imshow("Test OpenCV setup", dst);
		////等待键盘任意键按下关闭此窗口
		//waitKey(0);

	#pragma endregion

	#pragma region Scalar Demo 03 给图片填色

		/*
		dst = Mat(src.size(), src.type());
		dst = Scalar(0, 25, 255);
		dst = src.clone();
		namedWindow("dst Image", CV_WINDOW_AUTOSIZE);
		imshow("dst Image",dst);
		*/

	#pragma endregion	

	#pragma region RevertColor Demo 04 反转灰度图

		////转为灰度图像
		//cvtColor(src, gray_src, CV_BGR2GRAY);

		////拿到Width 跟 Height
		////其实也可以不要
		//int width = gray_src.rows;
		//int height = gray_src.cols;

		////反色
		//for (int row = 0; row < height; row++)
		//{
		//	for (int col = 0; col < width; col++)
		//	{
		//		int gray = gray_src.at<uchar>(row, col);
		//		gray_src.at<uchar>(row, col) = 255 - gray;
		//	}
		//}
		//

		////显示图像
		//namedWindow("Revert Image", CV_WINDOW_AUTOSIZE);
		//imshow("Revert Image", gray_src);

	#pragma endregion

	#pragma region Revert3Channels Demo 05 反转图像
			
	//创建dst图像
	dst.create(src.size(), src.type());
	//循环获取src的单个像素点的BGR数值
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
	//反色
	bitwise_not(src, dst);
	namedWindow("dst", CV_WINDOW_AUTOSIZE);
	imshow("dst", dst);
	*/

	#pragma endregion

	#pragma region Remix Image Demo 06 混合图像
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

	#pragma region Contrast Demo 07 调整亮度

	/*
	//先建立dst
	dst = Mat::zeros(src.size(), src.type());
	//设定alpha β 这里直接提高精度
	float alpha = 1.2;
	float beta = 10;

	//将src转为更高精度的Vec3F
	Mat m1;
	src.convertTo(m1, CV_32F);

	//读取src像素的bgr
	for (int row = 0; row < src.rows; row++)
	{
		for (int  col = 0; col < src.cols; col++)
		{
			//读取三通道的BGR值
			float b= m1.at<Vec3f>(row, col)[0];
			float g = m1.at<Vec3f>(row, col)[1];
			float r = m1.at<Vec3f>(row, col)[2];

			//设置dst的像素
			dst.at<Vec3b>(row, col)[0] = saturate_cast<uchar>(b*alpha + beta);
			dst.at<Vec3b>(row, col)[1] = saturate_cast<uchar>(g*alpha + beta);
			dst.at<Vec3b>(row, col)[2] = saturate_cast<uchar>(r*alpha + beta);
		}
	}
	imshow("dst", dst);

	*/


#pragma endregion

	#pragma region Draw LinesText Demo 8 画图

	//画直线
	/*
	Point p1 = Point(20, 30);
	Point p2 = Point(50, 70);
	Scalar color = Scalar(0, 0, 255);
	line(src, p1, p2, color, 2, LINE_8);
	imshow("src", src);
	*/

	//画矩形
	/*
	Rect rec = Rect(200, 100, 300, 300);
	Scalar color = Scalar(0, 255, 255);
	rectangle(src, rec, color, 2, LINE_4);
	*/

	//画椭圆
	/*
	ellipse(src, Point(src.cols / 2, src.rows / 2),Size(src.cols/4,src.rows/8),90,50,180, Scalar(0, 255, 255),2, LINE_4);
	imshow("src", src);
	*/

	//画圆
	//circle(src, Point(200, 200), 50, Scalar(0, 255, 255), 2, LINE_4);

	//写字
	//putText(src, "Hellow", Point(200, 200), CV_FONT_HERSHEY_DUPLEX,1.0, Scalar(255, 15, 255), 1, 8);
	
	//imshow("src", src);

	#pragma endregion

	#pragma region Random Draw Lines 9 随机画线
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

	#pragma region 模糊与高斯模糊
	
	
	
	/*
	blur(src, dst, Size(1, 155), Point(-1, -1));	
	GaussianBlur(src, dst, Size(11, 11), 5, 5);

	imshow("GaussianBlur", dst);
	*/

	#pragma endregion

	#pragma region 中值滤波

	//medianBlur(src, dst, 33);
	//bilateralFilter(src, dst, 15, 400,333);
	//imshow("src", dst);

	
	
	#pragma endregion

	//模糊
	/*Mat element = getStructuringElement(MORPH_RECT, Size(15, 15));
	erode(src, dst, element);
	imshow("ddd", dst);
	//namedWindow("Output_Img", CV_WINDOW_AUTOSIZE);
	//createTrackbar("Change It!", "Output_Img", &ele_size, 23, Tracebar_CallBack);
	*/

	//调用摄像头
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

	//形态学操作
	/*
	Mat kernel = getStructuringElement(MORPH_RECT, Size(11, 11), Point(-1, -1));
	morphologyEx(src, dst, CV_MOP_BLACKHAT, kernel);

	imshow("ex", dst);
	*/
		
	//提取横线、竖线
	/*
	Mat gary_img;
	cvtColor(src, gary_img, CV_BGR2GRAY);
	
	Mat bin_img;
	adaptiveThreshold(~gary_img, bin_img, 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY, 15, 0);
	imshow("bing", bin_img);
	
	//提出横向线，砍掉vertical线
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
	//归一化
	
	
	normalize(finTemp, dst, 255, 64, NORM_MINMAX);
	imshow("subs", dst);
	*/

	//BinaryImage
	/*
	namedWindow("OUT", CV_WINDOW_AUTOSIZE);
	
	createTrackbar("Adjust:", "OUT", &ele_size, 255, Trackbar_CallBack);
	Trackbar_CallBack(0, 0);
	*/

	#pragma region Sobel算子
	
	//顶部声明了src dst gray_src src1 src2
	
	//GaussianBlur(src, dst, Size(3, 3), 0, 0);
	//cvtColor(dst, gray_src, CV_BGR2GRAY);
	
	//边缘得到了更大的加强
	//Scharr(gray_src, src1, CV_16S, 1, 0);
	//Scharr(gray_src, src2, CV_16S, 0, 1);

	//Sobel(gray_src, src1, CV_16S, 1, 0, 3);
	//Sobel(gray_src, src2, CV_16S, 0, 1, 3);

	//convertScaleAbs(src1, src1);
	//convertScaleAbs(src2, src2);
	//imshow("x", src1);
	//imshow("y", src2);
	
	//图像混合
	//addWeighted(src1, 0.5, src2, 0.5, 0, dst);

	//自己写一个混合算法
	//不能使用dst，因为dst在此之前已经规定了了Mat格式
	
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
	
	#pragma region Laplacian 拉普拉斯算子

	/*
	GaussianBlur(src, dst, Size(3, 3), 0, 0);
	cvtColor(dst, gray_src, CV_BGR2GRAY);
	threshold(gray_src, gray_src,0, 255, THRESH_OTSU|THRESH_BINARY);
	Laplacian(gray_src, dst,CV_16S, 3);
	convertScaleAbs(dst, dst);


	imshow("output", dst);
	*/

	#pragma endregion
	
	#pragma region Canny 边缘检测

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
	
	#pragma region HoughLinsP 霍夫直线变换
	/*

	//先进行Canny边缘检测
	Mat edge_src;
	vector<Vec4f> plines;

	Canny(src, edge_src, 100, 200);
	//将边缘进行直线变换
	HoughLinesP(edge_src, plines, 1, CV_PI / 180.0, 5);
	//划线
	for (size_t i = 0; i < plines.size(); i++)
	{
		Vec4f hlines = plines[i];
		line(src, Point(hlines[0], hlines[1]), Point(hlines[2], hlines[3]), Scalar(15,255,59), 10);
	}
	imshow("OUT", src);
	*/

	#pragma endregion

	#pragma region HoughCircle 霍夫圆检测
	
	//函数说明
	/* --来自CSDN--

	第四个参数，double类型的dp，用来检测圆心的累加器图像的分辨率于输入图像之比的倒数，且此参数允许创建一个比输入图像分辨率低的累加器。上述文字不好理解的话，来看例子吧。例如，如果dp= 1时，累加器和输入图像具有相同的分辨率。如果dp=2，累加器便有输入图像一半那么大的宽度和高度。

	第五个参数，double类型的minDist，为霍夫变换检测到的圆的圆心之间的最小距离，即让我们的算法能明显区分的两个不同圆之间的最小距离。这个参数如果太小的话，多个相邻的圆可能被错误地检测成了一个重合的圆。反之，这个参数设置太大的话，某些圆就不能被检测出来了。

	第六个参数，double类型的param1，有默认值100。它是第三个参数method设置的检测方法的对应的参数。对当前唯一的方法霍夫梯度法CV_HOUGH_GRADIENT，它表示传递给canny边缘检测算子的高阈值，而低阈值为高阈值的一半。

	第七个参数，double类型的param2，也有默认值100。它是第三个参数method设置的检测方法的对应的参数。对当前唯一的方法霍夫梯度法CV_HOUGH_GRADIENT，它表示在检测阶段圆心的累加器阈值。它越小的话，就可以检测到更多根本不存在的圆，而它越大的话，能通过检测的圆就更加接近完美的圆形了。

	第八个参数，int类型的minRadius,有默认值0，表示圆半径的最小值。

	第九个参数，int类型的maxRadius,也有默认值0，表示圆半径的最大值。需要注意的是，使用此函数可以很容易地检测出圆的圆心，但是它可能找不到合适的圆半径
	*/
	
	/*

	//先进行滤波 这儿选用中值滤波
	medianBlur(src, dst, 3);
	cvtColor(dst,gray_src, CV_BGR2GRAY);
	//霍夫圆检测
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

	#pragma region  HistogramCalculate 直方图计算

	/*

	//看着头疼，没啥暖用
	//分通道显示
	vector<Mat> bgrSrc;
	split(src, bgrSrc);

	//calcHist
	calcHist(bgrSrc,)

	*/

	#pragma endregion
	
	#pragma region TemplateMatch 模板匹配
	
	//把下面的代码封装成MatchingMethod就可以了
	// createTrackbar( trackbar_label, image_window, &match_method, max_Trackbar, MatchingMethod );
	//MatchingMethod(0, 0);

	/*
	double minVal, maxVal;
	//模板匹配对于SQDIFF和SQDIFF_NORMED，越小的数值有着更高的匹配效果，其他的方法数值越大的匹配效果越好
	matchTemplate(src, baseT, dst, CV_TM_SQDIFF_NORMED);
	normalize(dst, dst, 0, 1, NORM_MINMAX);

	Point minLoc, maxLoc, matchLoc;
	//minMaxLoc在图像上找到最大值和最小值，并且存放在minLoc和maxLoc中
	minMaxLoc(dst, &minVal, &maxVal, &minLoc, &maxLoc, Mat());
	rectangle(src, minLoc, Point(minLoc.x + baseT.cols, minLoc.y + baseT.rows), Scalar(0, 255, 0));
	cout << "匹配度：" << minVal << endl;
	imshow("wht", src);
	*/


	#pragma endregion

	#pragma region FindContours 轮廓发现
	/*
	cvtColor(src, gray_src, CV_BGR2GRAY);
	namedWindow("out", WINDOW_AUTOSIZE);
	//此时已有灰度图像，canny_threshold默认为3
	createTrackbar("track", "out", &canny_threshold, 500, AdjustThreshold);
	//需要注意下要在前面声明
	AdjustThreshold(0, 0);
	imshow("out", dst);
	*/
	#pragma endregion

	#pragma region ConvexHull 凸包

	/*
	namedWindow("out", WINDOW_AUTOSIZE);
	cvtColor(src, gray_src, CV_BGR2GRAY);
	blur(gray_src, gray_src, Size(3, 3));
	createTrackbar("binary","out", &binary_threshold, 255, binary_callback);
	binary_callback(0, 0);
	imshow("out", dst);
	*/

	#pragma endregion

	#pragma region DrawPoly轮廓多边形绘制

	/*
	namedWindow("out", WINDOW_AUTOSIZE);
	cvtColor(src, gray_src, CV_BGR2GRAY);
	blur(gray_src, gray_src, Size(3, 3));
	createTrackbar("binary", "out", &binary_threshold, 255, Contours_DrawPoly);
	Contours_DrawPoly(0, 0);
	imshow("out", src);
	*/
	#pragma endregion

	#pragma region 图像分割

		//namedWindow("dst", WINDOW_AUTOSIZE);

		#pragma region 变成黑色背景
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
		//此时已是黑色背景
		//imshow("black image", src);
		#pragma endregion	

		#pragma region 锐化

		//先整一个拉普拉斯算子
		Mat kernel = (Mat_<float>(3, 3) << 1, 1, 1, 1, -8, 1, 1, 1, 1);
		filter2D(src, laplance_src, CV_32F, kernel);
		//imshow("lap", laplance_src);


		//32F的图像存入sharpImg
		src.convertTo(sharpImg, CV_32F);

		//这里要用上面黑色背景的图像（black src）不然出来结果不对
		sharpImg = sharpImg - laplance_src;
		sharpImg.convertTo(sharpImg, CV_8UC3);
		//imshow("AfterSharp", sharpImg);

		*/
		#pragma endregion

		#pragma region 二值化 距离变化 归一化
		/*
		//现在得到的sharpImg是8UC3格式 src仍然是black src dst暂时还没用上 Laplance_src后面可能用不上了
		//准备进行第三步 二值化
		//先转为灰色图像
		cvtColor(sharpImg, sharpImg, CV_BGR2GRAY);
		//进行二值化（其实40效果并不好，我觉得140可以）（他这加了个THRESH_OTSU，结果也不一样，后来要注意一下）
		threshold(sharpImg, gray_src, 40, 255, THRESH_BINARY|THRESH_OTSU);
		//进行距离变换
		distanceTransform(gray_src, dst, DIST_L2, 3);
		//变换完了还看不出结果，但是归一化之后就有效果了
		normalize(dst, dst, 0, 1, NORM_MINMAX);
		//二值化得到大概标记（确实是很抽象的标记了）
		threshold(dst, dst, 0.4, 1, THRESH_BINARY);
		//腐蚀
		Mat kernel1 = Mat::ones(3, 3, CV_8UC1);
		erode(dst, dst, kernel1);


		
		dst.convertTo(dst,CV_8U);
		Mat marks = Mat::zeros(dst.size(), CV_32SC1);
		//发现轮廓

		//用于保存找到的轮廓
		vector<vector<Point>> contours;
		findContours(dst, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
		//给一个随机数种子 随机数用做随机颜色  随机颜色用于填补不同区域
		
		for (size_t i = 0; i < contours.size(); i++)
		{
			Scalar color = Scalar::all(static_cast<int>(i)+1);
			drawContours(marks, contours, i, color, -1);//-1为填充内部
		}
		circle(marks, Point(5, 5), 3, Scalar(255, 255, 255), -1);
		//展示轮廓
		imshow("mkr",marks* 1000);

		
		//开始分水岭变换
		watershed(src, marks);
		//不*1000就是全黑 *了以后背景就是白色的，cards就显示出来了
		imshow("water", marks * 1000);
		//下面几行代码没啥用，就是给你看看分水岭的效果
		Mat msk = Mat::zeros(marks.size(), CV_8UC1);
		marks.convertTo(msk, CV_8UC1);
		bitwise_not(msk, msk,Mat());
		imshow("mak", msk);

		
		//开始上色 用的是分水岭后的marks
		vector<Vec3b> colors;
		//准备一个随机颜色
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
	//自动寻找合适的阈值THRESH_TRIANGLE|THRESH_BINARY
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

//轮廓发现Callback
void AdjustThreshold(int, void*)
{
	//canny_threshold
	Mat canout;
	Canny(gray_src, canout, canny_threshold, canny_threshold * 2);
	//vector知识点见 https://blog.csdn.net/u010368556/article/details/79179669
	//hierachy 等级制度
	//用于保存找到的图像等级
	vector<Vec4i> hierachy;
	//用于保存找到的轮廓
	vector<vector<Point>> contours;
	
	//知识点见 https://blog.csdn.net/keith_bb/article/details/70185209
	//轮廓检索模式
	/*
	RETR_EXTERNAL:表示只检测最外层轮廓，对所有轮廓设置hierarchy[i][2]=hierarchy[i][3]=-1 
	RETR_LIST:提取所有轮廓，并放置在list中，检测的轮廓不建立等级关系 
	RETR_CCOMP:提取所有轮廓，并将轮廓组织成双层结构(two-level hierarchy),顶层为连通域的外围边界，次层位内层边界 
	RETR_TREE:提取所有轮廓并重新建立网状轮廓结构 
	RETR_FLOODFILL：官网没有介绍，应该是洪水填充法 
	*/	
	//轮廓近似方法
	/*
	CHAIN_APPROX_NONE：获取每个轮廓的每个像素，相邻的两个点的像素位置差不超过1 
	CHAIN_APPROX_SIMPLE：压缩水平方向，垂直方向，对角线方向的元素，值保留该方向的重点坐标，如果一个矩形轮廓只需4个点来保存轮廓信息 
	CHAIN_APPROX_TC89_L1和CHAIN_APPROX_TC89_KCOS使用Teh-Chinl链逼近算法中的一种
	*/
	findContours(canout, contours, hierachy, RETR_TREE, CHAIN_APPROX_SIMPLE);
	//给一个随机数种子 随机数用做随机颜色  随机颜色用于填补不同区域
	RNG rng(12345);
	for (size_t i = 0; i < contours.size(); i++)
	{
		Scalar color = Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
		drawContours(dst, contours, i, color, 2, LINE_AA, hierachy);
	}
	imshow("out", dst);
}

//凸包发现
void binary_callback(int, void*)
{
	Mat canout;


	//用于保存找到的图像等级
	vector<Vec4i> hierachy;
	//用于保存找到的轮廓
	vector<vector<Point>> contours;
	threshold(gray_src, canout, binary_threshold, 255, THRESH_BINARY);
	//发现轮廓
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

//绘制轮廓多边形
void Contours_DrawPoly(int, void*)
{
	Mat canout;
	//用于保存找到的图像等级
	vector<Vec4i> hierachy;
	//用于保存找到的轮廓
	vector<vector<Point>> contours;
	
	threshold(gray_src, canout, binary_threshold, 255, THRESH_BINARY);
	//发现轮廓
	findContours(canout, contours, hierachy, RETR_TREE, CHAIN_APPROX_SIMPLE);
	
	


	//存放矩形
	vector<Rect> rectangles(contours.size());
	//存放
	vector<Point2f> centers(contours.size());
	vector<float> radiux(contours.size());

	//减小点数
	vector<vector<Point>> contours_covs(contours.size());
	for (size_t i = 0; i < contours.size(); i++)
	{
		approxPolyDP(contours[i], contours_covs[i], 3, true);
		//把一列点整理成了一个矩形（而不是转成了一个点）
		rectangles[i] = boundingRect(contours_covs[i]);
		//把contours里的点整理成圆，输出圆的中心与半径
		minEnclosingCircle(contours_covs[i], centers[i], radiux[i]);

	}
	//绘制
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