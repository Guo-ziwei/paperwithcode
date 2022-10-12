#include "../include/nms.h"

#include <iostream>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace std;

int main(int argc, char const* argv[]) {
    NMS nms(5);
    if (argc != 2) {
        cout << "usage: feature_extraction img1" << endl;
        return 1;
    }
    //-- 读取图像
    cv::Mat img_1 = cv::imread(argv[1], cv::IMREAD_GRAYSCALE);
    std::vector<cv::KeyPoint> keypoints_1;
    cv::Mat descriptors_1;
    cv::Ptr<cv::FeatureDetector> detector = cv::ORB::create();
    cv::Ptr<cv::DescriptorExtractor> descriptor = cv::ORB::create();
    //-- 第一步:检测 Oriented FAST 角点位置
    chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
    detector->detect(img_1, keypoints_1);
    std::vector<cv::Point2f> cur_pts, result_pts;
    std::vector<int> queryidxs;
    for (size_t i = 0; i < keypoints_1.size(); i++) {
        queryidxs.push_back(i);
    }

    cv::KeyPoint::convert(keypoints_1, cur_pts, queryidxs);
    nms.run(img_1, cur_pts, result_pts);
    return 0;
}
