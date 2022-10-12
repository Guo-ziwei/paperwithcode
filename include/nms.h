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
    NMS(size_t window_size): window_size_(window_size), n_((window_size - 1) / 2) {}
    ~NMS() = default;
    template <typename Scalar>
    void run(
        const cv::Mat image, const std::vector<cv::Point_<Scalar>>& points,
        std::vector<cv::Point_<Scalar>>& result_points) {
        int begin_index_x{0}, begin_index_y{0};
        for (size_t i = n_; i < image.rows - n_; i += n_ + 1) {
            for (size_t j = n_; j < image.cols - n_; j += n_ + 1) {
                bool is_not_max = false;
                size_t mi = i, mj = j;
                for (size_t i2 = i; i2 < i + n_ + 1; i2++) {
                    for (size_t j2 = j; j2 < j + n_ + 1; j2++) {
                        if (image.at<u_char>(i2, j2) > image.at<u_char>(mi, mj)) {
                            mi = i2;
                            mj = j2;
                        }
                    }
                }
                for (size_t i2 = mi - n_; i2 < mi + n_ + 1; i2++) {
                    if (is_not_max)
                        break;
                    for (size_t j2 = mj - n_; j2 < mj + n_ + 1; j2++) {
                        if (!isInBoarder(cv::Point_<Scalar>(i2, j2), i, j, i + n_, j + n_)) {
                            if (image.at<u_char>(i2, j2) > image.at<u_char>(mi, mj)) {
                                is_not_max = true;
                                break;
                            }
                        }
                    }
                }
                if (!is_not_max)
                    result_points.push_back(cv::Point_<Scalar>(mi, mj));
            }
        }
    }
};
