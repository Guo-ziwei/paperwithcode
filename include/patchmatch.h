#pragma once

#include <cstddef>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <random>

namespace stereo {

class Plane {
  private:
    /* data */
    cv::Vec3f point_;
    cv::Vec3f normal_;
    cv::Vec3f coeff_;

  public:
    Plane() = default;
    Plane(cv::Vec3f point, cv::Vec3f normal_vec);  // Random Initialization
    ~Plane() = default;
    Plane viewTransform(int x, int y, int sign, int& qx, int& qy);

    const cv::Vec3f& getPoint() const;
    const cv::Vec3f& getNormal() const;
    const cv::Vec3f& getCoeff() const;
};

class PatchMatch {
  private:
    float alpha_, gamma_, tau_col_, tau_gard_;
    cv::Mat3b views[2];  // left and right view images
    cv::Mat2f grads[2];  // pixels greyscale gradient for both views
    cv::Mat1f disps[2];  // left and right disparity maps
    cv::Mat1f costs[2];  // planes's costs
    cv::Mat weights[2];  // precomputed pixel window weights
    int rows, cols;
    Plane **planes1, **planes2;
    float disSimilarity(
        const cv::Vec3f& q, const cv::Vec3f& q_, const cv::Vec2f& grad_q,
        const cv::Vec2f& grad_q_);  // function pho(q,q') equ(5)
    void preComputePixelsWeights(const cv::Mat3b& frame, cv::Mat& weights, int window_length);
    void computeGreyGradient(const cv::Mat3b& frame, cv::Mat2f& grad);
    void initializeRandomPlanes(Plane** planes, float max_d);
    void spatialPropagation(int x, int y, int image_view, int iter);
    void viewPropagation(int x, int y, int image_view);
    // void temporalPropagation(int x, int y);  // can only be used when working on stereo video
    void planeRefinement(
        int x, int y, int image_view, float max_delta_z, float max_delta_n, float end_delta_z);
    float planeMatchCost(const Plane& p, int cx, int cy, int window_length, int image_view);
    void fillInvalidPixels(int y, int x, Plane** planes, const cv::Mat1b& validity);
    void planesToDisparity(Plane** const planes, cv::Mat1f& disp);
    void weightedMedianFilter(
        int cx, int cy, cv::Mat1f& disparity, const cv::Mat& weights, const cv::Mat1b& valid,
        int window_length, bool use_invalid);

  public:
    PatchMatch(float alpha, float gamma, float tau_col, float tau_gard);
    PatchMatch& operator=(const PatchMatch& pm) = delete;
    ~PatchMatch();
    cv::Mat1f getLeftDisparityMap() const;
    cv::Mat1f getRightDisparityMap() const;
    void init(const cv::Mat3b& img1, const cv::Mat3b& img2);
    void propagation(int iteration, bool reverse = false);
    void postProcess();
};

inline float weight(const cv::Vec3f& p, const cv::Vec3f& q, float gamma) {
    return std::exp(-cv::norm(p - q, cv::NORM_L1) / gamma);
}  // equ(4)

inline bool isInside(
    int x, int y, int lowbound_x, int lowbound_y, int upperbound_x, int upperbound_y) {
    return lowbound_x <= x && x < upperbound_x && lowbound_y <= y && y < upperbound_y;
}

inline float disparity(float x, float y, const Plane& p) {
    const cv::Vec3f& coeff = p.getCoeff();
    return coeff[0] * x + coeff[1] * y + coeff[2];
}

template <typename T>
inline T linearInterpolation(const T& x, const T& y, float wx) {
    return wx * x + (1 - wx) * y;
}

}  // namespace stereo