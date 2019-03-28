#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include "gms_matcher.h"
using namespace std;
using namespace cv;

Mat DrawInlier(Mat &src1, Mat &src2, vector<KeyPoint> &kpt1, vector<KeyPoint> &kpt2, vector<DMatch> &inlier, int type);
int main(void)
{
    clock_t startTime,endTime;
	// 配置图像路径
	Mat img1 = imread("/home/lee/Desktop/dataset/train/bikes1.ppm");
	Mat img2 = imread("/home/lee/Desktop/dataset/train/bikes2.ppm");
	// 判断输入图像是否读取成功
    startTime = clock();
    OrbFeatureDetector orb(1000);
	// 调整精度，值越小点越少，越精准
	vector<KeyPoint> kpts1, kpts2;
	// 特征点检测算法...
	orb.detect(img1, kpts1);
	orb.detect(img2, kpts2);


	// 特征点描述算法...
	Mat desc1, desc2;
	bool SelectiveDescMethods = true;
	// 默认选择BRIEF描述符
	// ORB 算法中默认BRIEF描述符
	orb.compute(img1, kpts1, desc1);
	orb.compute(img2, kpts2, desc2);
    // 粗精匹配数据存储结构
    vector< vector<DMatch>> matches;
    vector<DMatch> goodMatchKpts;
    // Keypoint Matching...
    DescriptorMatcher *pMatcher = new BFMatcher(NORM_HAMMING, false);
    pMatcher->knnMatch(desc1, desc2, matches, 2);
    // 欧式距离度量  阈值设置为0.8
    for (unsigned int i = 0; i < matches.size(); ++i)
    {
        if (matches[i][0].distance < 0.8*matches[i][1].distance)
        {
            goodMatchKpts.push_back(matches[i][0]);
        }
    }
    cout << "初次匹配结点：" <<goodMatchKpts.size()<< endl;
    // 显示匹配点对
   /* Mat show_match;
    drawMatches(img1, kpts1, img2, kpts2, goodMatchKpts, show_match);

    imshow("ORB_Algorithms_", show_match);

    cout << "(kpts1: " << kpts1.size() << ") && (kpts2:" \
         << kpts2.size() << ") = goodMatchesKpts: " << goodMatchKpts.size() << endl;

    waitKey(0);
    // RANSAC Geometric Verification
    if (goodMatchKpts.size() < 4)
    {
        cout << "The Match Kpts' Size is less than Four to estimate!" << endl;
        return 0;
    }
    //Ransac
    vector<Point2f> obj, scene;
    for (unsigned int i = 0; i < goodMatchKpts.size(); ++i)
    {
        obj.push_back(kpts1[goodMatchKpts[i].queryIdx].pt);
        scene.push_back(kpts2[goodMatchKpts[i].trainIdx].pt);
    }
    // 估计Two Views变换矩阵
    Mat H = findHomography(obj, scene, CV_RANSAC);
    vector<Point2f> obj_corners(4), scene_corners(4);
    obj_corners[0] = cvPoint(0, 0); obj_corners[1] = cvPoint(img1.cols, 0);
    obj_corners[2] = cvPoint(img1.cols, img1.rows); obj_corners[3] = cvPoint(0, img1.rows);
    // 点集变换标出匹配重复区域
    perspectiveTransform(obj_corners, scene_corners, H);

    line(show_match, scene_corners[0] + Point2f(img1.cols, 0), scene_corners[1] + Point2f(img1.cols, 0), Scalar(0, 255, 0), 4);
    line(show_match, scene_corners[1] + Point2f(img1.cols, 0), scene_corners[2] + Point2f(img1.cols, 0), Scalar(0, 255, 0), 4);
    line(show_match, scene_corners[2] + Point2f(img1.cols, 0), scene_corners[3] + Point2f(img1.cols, 0), Scalar(0, 255, 0), 4);
    line(show_match, scene_corners[3] + Point2f(img1.cols, 0), scene_corners[0] + Point2f(img1.cols, 0), Scalar(0, 255, 0), 4);

    imshow("Match End", show_match);
   // imwrite("img_boat15.jpg", show_match);
    waitKey(0);
    system("pause");
    return 0;*/

    vector<DMatch> m_Matches;
    m_Matches = goodMatchKpts;
    int ptCount = goodMatchKpts.size();
   /* if (ptCount < 100)
    {
        cout << "Don't find enough match points" << endl;
        return 0;
    }*/

    //坐标转换为float类型
    vector <KeyPoint> RAN_KP1, RAN_KP2;
    //size_t是标准C库中定义的，应为unsigned int，在64位系统中为long unsigned int,在C++中为了适应不同的平台，增加可移植性。
    for (size_t i = 0; i < m_Matches.size(); i++)
    {
        RAN_KP1.push_back(kpts1[goodMatchKpts[i].queryIdx]);
        RAN_KP2.push_back(kpts2[goodMatchKpts[i].trainIdx]);
        //RAN_KP1是要存储img01中能与img02匹配的点
        //goodMatches存储了这些匹配点对的img01和img02的索引值
    }
    //坐标变换
    vector <Point2f> p01, p02;
    for (size_t i = 0; i < m_Matches.size(); i++)
    {
        p01.push_back(RAN_KP1[i].pt);
        p02.push_back(RAN_KP2[i].pt);
    }
    vector<uchar> RansacStatus;
    Mat Fundamental = findFundamentalMat(p01, p02, RansacStatus, FM_RANSAC);
    //重新定义关键点RR_KP和RR_matches来存储新的关键点和基础矩阵，通过RansacStatus来删除误匹配点
    vector <KeyPoint> RR_KP1, RR_KP2;
    vector <DMatch> RR_matches;
    int index = 0;
    for (size_t i = 0; i < m_Matches.size(); i++)
    {
        if (RansacStatus[i] != 0)
        {
            RR_KP1.push_back(RAN_KP1[i]);
            RR_KP2.push_back(RAN_KP2[i]);
            m_Matches[i].queryIdx = index;
            m_Matches[i].trainIdx = index;
            RR_matches.push_back(m_Matches[i]);
            index++;
        }
    }
    endTime = clock();
    cout << "orb-ransac时间:"<<(double)(endTime - startTime) / CLOCKS_PER_SEC << endl;
    cout << "RANSAC后匹配点数" <<RR_matches.size()<< endl;

   /* Mat img_RR_matches;
    drawMatches(img1, RR_KP1, img2, RR_KP2, RR_matches, img_RR_matches);
    imshow("After RANSAC",img_RR_matches);*/

    Mat show1 = DrawInlier(img1, img2, RR_KP1, RR_KP2, RR_matches, 1);
    imshow("orb", show1);
   // imwrite("/home/lee/Desktop/dataset/expriementResult/3_orb.jpg", show1);
    //等待任意按键按下
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
