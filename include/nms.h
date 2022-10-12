#pragma once

#include <iostream>
#include <opencv2/core/core.hpp>
#include <vector>

class NMS {
  private:
    size_t window_size_;  // 2 * n + 1
    size_t n_;
    template <typename Scalar>
    bool isInBoarder(
        const cv::Point_<Scalar>& points, size_t top_left_x, size_t top_left_y,
        size_t bottom_right_x, size_t bottom_right_y) {
        if (points.x >= top_left_x && points.y >= top_left_y && points.x <= bottom_right_x &&
            points.y <= bottom_right_y) {
            return true;
        }
        return false;
    }

  public:
    NMS() = delete;
    NMS(size_t window_size);
    ~NMS() = default;
    template <typename Scalar>
    void run(
        const cv::Mat image, const std::vector<cv::Point_<Scalar>>& points,
        std::vector<cv::Point_<Scalar>>& result_points) {
        int begin_index_x{0}, begin_index_y{0};
        for (size_t i = n_; i < image.rows - n_; i += n_) {
            for (size_t j = n_; j < image.cols - n_; j += n_) {
                size_t mi = i, mj = j;
                for (size_t i2 = i; i2 < i + n; i2++) {
                    for (size_t j2 = j; j2 < j + n; j2++) {
                        if (image.at<u_char>(i2, j2) > image.at<u_char>(mi, mj)) {
                            mi = i2;
                            mj = j2;
                        }
                    }
                }
                for (size_t i2 =  i; i2 < mi - n; i2++) {
                    for (size_t j2 = j; j2 < mj - n; j2++){}
                }
            }
        }
    }
};
