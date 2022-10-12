#include "../include/nms.h"

#include <iostream>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace std;

int main(int argc, char const* argv[]) {
    NMS nms(25);
    if (argc != 2) {
        cout << "usage: feature_extraction img1" << endl;
        return 1;
    }
    //-- 读取图像
    cv::Mat img_1 = cv::imread(argv[1], cv::IMREAD_GRAYSCALE);
    std::cout << "image size: " << img_1.rows << " " << img_1.cols << std::endl;
    std::vector<cv::KeyPoint> keypoints_1;
    cv::Mat descriptors_1;
    cv::Ptr<cv::FeatureDetector> detector = cv::ORB::create();
    cv::Ptr<cv::DescriptorExtractor> descriptor = cv::ORB::create();
    //-- 第一步:检测 Oriented FAST 角点位置
    detector->detect(img_1, keypoints_1);
    std::vector<cv::Point2f> cur_pts, result_pts;
    std::vector<int> queryidxs;
    for (size_t i = 0; i < keypoints_1.size(); i++) {
        queryidxs.push_back(i);
    }

    cv::KeyPoint::convert(keypoints_1, cur_pts, queryidxs);
    nms.run(img_1, cur_pts, result_pts);
    for (const auto& pts : result_pts) {
        //  src.at(i,j) is using (i,j) as (row,column) but Point(x,y) is using (x,y) as (column,row)
        cv::circle(img_1, cv::Point2f(pts.y, pts.x), 2, cv::Scalar(0, 0, 250), 2);
    }
    std::cout << "total pts: " << result_pts.size() << std::endl;
    cv::imshow("nms", img_1);
    cv::waitKey();
    return 0;
}
