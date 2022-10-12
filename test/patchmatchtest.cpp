#include <iostream>
#include <sys/time.h>

#include "../include/patchmatch.h"

int main(int argc, char const* argv[]) {
    /* code */
    struct timeval start, end;
    const float alpha = 0.9f;
    const float gamma = 10.0f;
    const float tau_col = 10.0f;
    const float tau_gard = 2.0f;

    cv::Mat3b img1 = cv::imread(argv[1], cv::IMREAD_COLOR);
    cv::Mat3b img2 = cv::imread(argv[2], cv::IMREAD_COLOR);
    stereo::PatchMatch patchmatch(alpha, gamma, tau_col, tau_gard);
    gettimeofday(&start, nullptr);
    patchmatch.init(img1, img2);
    gettimeofday(&end, nullptr);
    double elapsed_seconds = (end.tv_sec - start.tv_sec) * 1e3;
    elapsed_seconds += (end.tv_usec - start.tv_usec) * 1e-3;
    std::cout << "init elapsed time: " << elapsed_seconds << std::endl;
    gettimeofday(&start, nullptr);
    patchmatch.propagation(3);
    gettimeofday(&end, nullptr);
    elapsed_seconds = (end.tv_sec - start.tv_sec) * 1e3;
    elapsed_seconds += (end.tv_usec - start.tv_usec) * 1e-3;
    std::cout << "propagation elapsed time: " << elapsed_seconds << std::endl;
    gettimeofday(&start, nullptr);
    patchmatch.postProcess();
    gettimeofday(&end, nullptr);
    elapsed_seconds = (end.tv_sec - start.tv_sec) * 1e3;
    elapsed_seconds += (end.tv_usec - start.tv_usec) * 1e-3;
    std::cout << "post process elapsed time: " << elapsed_seconds << std::endl;
    cv::Mat1f disp1 = patchmatch.getLeftDisparityMap();
    cv::Mat1f disp2 = patchmatch.getRightDisparityMap();
    cv::normalize(disp1, disp1, 0, 255, cv::NORM_MINMAX);
    cv::normalize(disp2, disp2, 0, 255, cv::NORM_MINMAX);
    cv::imwrite("left_disparity.png", disp1);
    cv::imwrite("right_disparity.png", disp2);
    return 0;
}
