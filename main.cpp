#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/nonfree/nonfree.hpp>

using namespace std;
using namespace cv;

class a : public CV_EXPORTS_W GridAdaptedFeatureDetector
        {

public:
    a (const Ptr<FeatureDetector>& detector= 0,
       int maxTotalKeypoints=1000,
       int gridRows=5, int gridCols=5 )
    {
        cout << gridCols << gridRows << endl;
    }
};


int main() {
    initModule_nonfree();
    clock_t startTime,endTime;

    Mat img = imread("/home/lee/Desktop/dataset/train/bikes1.ppm",0);

    vector<KeyPoint> kpts1,kpts2;

    const string  detectorType = "GridFAST";

    CV_EXPORTS_W GridAdaptedFeatureDetector de(FeatureDetector::create(detectorType),1000,4,4);
    //CV_EXPORTS_W GridAdaptedFeatureDetector de2(FeatureDetector::create(detectorType),2000,10,10);
  //  Ptr<FeatureDetector> de2 = FeatureDetector::create("FAST");
    OrbFeatureDetector OrbDetector(1000);
    startTime = clock();//计时开始
    de.detect(img ,kpts1,Mat());
    endTime = clock();//计时结束

    cout << "第一次"<<(double)(endTime - startTime) / CLOCKS_PER_SEC << endl;
    startTime = clock();//计时开始
  //  de2.detect(img ,kpts2,Mat());
    OrbDetector.detect(img, kpts2);
    endTime = clock();//计时结束

    cout << "第二次"<<(double)(endTime - startTime) / CLOCKS_PER_SEC << endl;
  //  cout << "DFD " ;




//
//    Ptr<FeatureDetector> detect2D = FeatureDetector::create(detectorType);
//    detect2D -> detect(img ,kpts1,Mat());

    Mat outImageGrid,outImageGrid2 ,outImage;

    drawKeypoints(img, kpts1,outImageGrid ,Scalar(0,0,255));
    drawKeypoints(img, kpts2,outImageGrid2 ,Scalar(0,0,255));

    cout << kpts1.size() << "  " << kpts2.size() <<endl;

    namedWindow("GridFast");
    imshow("GridFast", outImageGrid);
    namedWindow("Fast");
    imshow("Fast", outImageGrid2);

  //  imwrite("/home/lee/Desktop/dataset/bikes/Gridfast1.jpg",outImageGrid);
   // imwrite("/home/lee/Desktop/dataset/bikes/fast1.jpg",outImageGrid2);
    waitKey(0);
   // std::cout << "Hello, World!" << std::endl;
    return 0;
}