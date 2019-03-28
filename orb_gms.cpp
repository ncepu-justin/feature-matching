#include "gms_matcher.h"

//#define USE_GPU 
#ifdef USE_GPU
#include <opencv2/cudafeatures2d.hpp>
using cuda::GpuMat;
#endif

void GmsMatch(Mat &img1, Mat &img2);
Mat DrawInlier(Mat &src1, Mat &src2, vector<KeyPoint> &kpt1, vector<KeyPoint> &kpt2, vector<DMatch> &inlier, int type);

void runImagePair() {
	Mat img1 = imread("/home/lee/Desktop/dataset/train/2.png");
	Mat img2 = imread("/home/lee/Desktop/dataset/train/3.png");

	GmsMatch(img1, img2);
}


int main()
{
#ifdef USE_GPU
	int flag = cuda::getCudaEnabledDeviceCount();
	if (flag != 0) { cuda::setDevice(0); }
#endif // USE_GPU

	runImagePair();

	return 0;
}

void GmsMatch(Mat &img1, Mat &img2) {
	vector<KeyPoint> kp1, kp2;
	clock_t startTime,endTime;

	Mat d1, d2;
	vector<DMatch> matches_all1,matches_gms1;
	/*const string  detectorType1 = "GridFAST";
	const string  detectorType2 = "PyramidFAST";
1
	CV_EXPORTS_W GridAdaptedFeatureDetector de(FeatureDetector::create(detectorType1),5000,5,5);
	//CV_EXPORTS_W GridAdaptedFeatureDetector de2(FeatureDetector::create(detectorType),5000,5,5);
	Ptr<FeatureDetector> detect2D = FeatureDetector::create(detectorType2);
    detect2D -> detect(img1 ,kp1,Mat());
    detect2D -> detect(img2 ,kp2,Mat());
	cout << "使用金字塔fast"<<kp1.size() << endl;
	cout << "使用金字塔fast"<<kp2.size() << endl;*/

	/*startTime = clock();//计时开始
	de.detect(img1 ,kp1,Mat());
	endTime = clock();//计时结束
	cout << "使用网格fast"<<kp1.size() << endl;
	//使用网格fast提取特征点
	cout << "第一次"<<(double)(endTime - startTime) / CLOCKS_PER_SEC << endl;
	startTime = clock();//计时开始
	de.detect(img2 ,kp2,Mat());
	endTime = clock();//计时结束
	cout << "使用网格fast"<<kp2.size() << endl;
	cout << "第二次"<<(double)(endTime - startTime) / CLOCKS_PER_SEC << endl;*/

//	Ptr<ORB> orb = ORB::create("10000");bInliers1
    //使用ORB特征提取
    startTime = clock();
	OrbFeatureDetector OrbDetector(150);
	OrbDescriptorExtractor OrbDescriptor;

	//cout <<" d" << endl;
	OrbDetector.detect(img1, kp1);
	OrbDetector.detect(img2, kp2);
	OrbDescriptor.compute(img1,kp1,d1);
	OrbDescriptor.compute(img2,kp2,d2);
//	OrbDescriptor.compute(img1,kp1,d1);
//	OrbDescriptor.compute(img2,kp2,d2);

#ifdef USE_GPU
	GpuMat gd1(d1), gd2(d2);
	Ptr<cuda::DescriptorMatcher> matcher = cv::cuda::DescriptorMatcher::createBFMatcher(NORM_HAMMING);
	matcher->match(gd1, gd2, matches_all);
#else
	BFMatcher matcher(NORM_HAMMING);
	matcher.match(d1, d2, matches_all1);
//	matcher.match(d3, d4, matches_all2);
#endif
	// GMS filter
	std::vector<bool> vbInliers1;
	std::vector<bool> vbInliers2;
	gms_matcher gms(kp1, img1.size(), kp2, img2.size/*	//LDB描述子描述FAsT特征点
		bool flag = true;
		LDB ldb(48);
		ldb.compute(img1, kpts1, desc1, flag);
		ldb.compute(img2, kpts2, desc2, flag);*/(), matches_all1);
//	gms_matcher gms2(kp3, img1.size(), kp4, img2.size(), matches_all2);

	int num_inliers = gms.GetInlierMask(vbInliers1, false, false);
	//int num_inliers2 = gms2.GetInlierMask(vbInliers2, false, false);
//	cout << "Get total " << num_inliers  << " matches." << endl;

	// collect matches
	for (size_t i = 0; i < vbInliers1.size(); ++i)
	{
		if (vbInliers1[i] == true)
		{
			matches_gms1.push_back(matches_all1[i]);
		}
	}
    endTime = clock();
    cout << "gms:"<<(double)(endTime - startTime) / CLOCKS_PER_SEC << endl;
	cout << "Get total " << matches_gms1.size()   << " matches." << endl;

	// draw matching
	Mat show1 = DrawInlier(img1, img2, kp1, kp2, matches_gms1, 1);
	imshow("show", show1);
	imwrite("/home/lee/Desktop/dataset/expriementResult/application_gms.jpg", show1);
	/*Mat img_RR_matches;
	drawMatches(img1, kp1, img2, kp2, matches_gms1, img_RR_matches);
	imshow("orb_gms",img_RR_matches);*/
	waitKey(0);
}

Mat DrawInlier(Mat &src1, Mat &src2, vector<KeyPoint> &kpt1, vector<KeyPoint> &kpt2, vector<DMatch> &inlier, int type) {
	const int height = max(src1.rows, src2.rows);
	const int width = src1.cols + src2.cols;
	Mat output(height, width, CV_8UC3, Scalar(0, 0, 0));
	src1.copyTo(output(Rect(0, 0, src1.cols, src1.rows)));
	src2.copyTo(output(Rect(src1.cols, 0, src2.cols, src2.rows)));

	if (type == 1)
	{
		for (size_t i = 0; i < inlier.size(); i++)
		{
			Point2f left = kpt1[inlier[i].queryIdx].pt;
			Point2f right = (kpt2[inlier[i].trainIdx].pt + Point2f((float)src1.cols, 0.f));
			line(output, left, right, Scalar(0, 255, 255));
		}
	}
	else if (type == 2)
	{
		for (size_t i = 0; i < inlier.size(); i++)
		{
			Point2f left = kpt1[inlier[i].queryIdx].pt;
			Point2f right = (kpt2[inlier[i].trainIdx].pt + Point2f((float)src1.cols, 0.f));
			line(output, left, right, Scalar(255, 0, 0));
		}

		for (size_t i = 0; i < inlier.size(); i++)
		{
			Point2f left = kpt1[inlier[i].queryIdx].pt;
			Point2f right = (kpt2[inlier[i].trainIdx].pt + Point2f((float)src1.cols, 0.f));
			circle(output, left, 1, Scalar(0, 255, 255), 2);
			circle(output, right, 1, Scalar(0, 255, 0), 2);
		}
	}

	return output;
}
