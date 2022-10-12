#include "../include/patchmatch.h"

#include <omp.h>

#define WINDOW_SIZE 35
#define MAX_DISPARITY 60
#define PLANE_PENALTY 120

namespace stereo {
Plane::Plane(cv::Vec3f point, cv::Vec3f normal_vec) : point_(point), normal_(normal_vec) {
    float a = -normal_vec[0] / normal_vec[2];
    float b = -normal_vec[1] / normal_vec[2];
    float c = cv::sum(normal_vec.mul(point))[0] / normal_vec[2];
    coeff_ = cv::Vec3f(a, b, c);
}

const cv::Vec3f& Plane::getPoint() const {
    return point_;
}

const cv::Vec3f& Plane::getNormal() const {
    return normal_;
}

const cv::Vec3f& Plane::getCoeff() const {
    return coeff_;
}

Plane Plane::viewTransform(int x, int y, int sign, int& qx, int& qy) {
    float z = coeff_[0] * x + coeff_[1] * y + coeff_[2];
    qx = x + sign * z;
    qy = y;
    cv::Vec3f p(qx, qy, z);
    return Plane(p, normal_);
}

PatchMatch::PatchMatch(float alpha, float gamma, float tau_col, float tau_gard)
    : alpha_(alpha), gamma_(gamma), tau_col_(tau_col), tau_gard_(tau_gard) {}

PatchMatch::~PatchMatch() {
    if (planes1) {
        for (int i = 0; i < rows; i++) {
            delete[] planes1[i];
            delete[] planes2[i];
        }
        delete[] planes1;
        delete[] planes2;
    }
}

void PatchMatch::computeGreyGradient(const cv::Mat3b& frame, cv::Mat2f& grad) {
    cv::Mat gray, x_gard, y_gard;
    cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
    cv::Sobel(gray, x_gard, CV_32F, 1, 0, 3);
    cv::Sobel(gray, y_gard, CV_32F, 0, 1, 3);
    //除以8是为了让梯度的最大值不超过255，这样计算代价时梯度差和颜色差位于同一个尺度
    x_gard = x_gard / 8;
    y_gard = y_gard / 8;
#pragma omp parallel for num_threads(8) collapse(2)
    for (int y = 0; y < frame.rows; y++) {
        for (int x = 0; x < frame.cols; x++) {
            grad(y, x)[0] = x_gard.at<float>(y, x);
            grad(y, x)[1] = y_gard.at<float>(y, x);
        }
    }
}

float PatchMatch::disSimilarity(
    const cv::Vec3f& q, const cv::Vec3f& q_, const cv::Vec2f& grad_q, const cv::Vec2f& grad_q_) {
    return (1 - alpha_) * std::min(static_cast<float>(cv::norm(q - q_, cv::NORM_L1)), tau_col_) +
           alpha_ *
               std::min(static_cast<float>(cv::norm(grad_q - grad_q_, cv::NORM_L1)), tau_gard_);
}

void PatchMatch::initializeRandomPlanes(Plane** planes, float max_d) {
    std::random_device dev;
    std::mt19937 rng(dev());
    std::uniform_real_distribution<float> rand_d(0.0, max_d);
    std::uniform_real_distribution<float> rand_n(-1.0, 1.0);
#pragma omp parallel for num_threads(8) collapse(2)
    for (int y = 0; y < rows; y++) {
        /* code */
        for (int x = 0; x < cols; x++) {
            float z = rand_d(rng);  // random disparity
            if (z == 0.0) {
                z = rand_d(rng);
            }
            cv::Vec3f point(x, y, z);
            float nx = rand_n(rng);
            float ny = rand_n(rng);
            float nz = rand_n(rng);
            cv::Vec3f normal(nx, ny, nz);
            cv::normalize(normal, normal);
            planes[y][x] = Plane(point, normal);
        }
    }
}

void PatchMatch::spatialPropagation(int x, int y, int image_view, int iter) {
    std::vector<std::pair<int, int>> offsets;
    if (iter % 2 == 0) {
        // left and upper neighbors
        offsets.push_back(std::make_pair(-1, 0));
        offsets.push_back(std::make_pair(0, -1));
    } else {
        // right and lower neighbors
        offsets.push_back(std::make_pair(1, 0));
        offsets.push_back(std::make_pair(0, 1));
    }
    int sign = (image_view == 0) ? -1 : 1;  // -1 processing left, +1 processing right
    float& old_cost = costs[image_view](y, x);
    if (sign < 0) {
        Plane& old_plane = planes1[y][x];
        for (auto& offset : offsets) {
            int ny = y + offset.first;
            int nx = x + offset.second;
            if (!isInside(nx, ny, 0, 0, cols, rows))
                continue;
            const Plane& plane_p = planes1[ny][nx];
            float new_cost = planeMatchCost(plane_p, x, y, WINDOW_SIZE, image_view);
            if (new_cost < old_cost) {
                old_plane = plane_p;
                old_cost = new_cost;
            }
        }
    } else {
        Plane& old_plane = planes2[y][x];
        for (auto& offset : offsets) {
            int ny = y + offset.first;
            int nx = x + offset.second;
            if (!isInside(nx, ny, 0, 0, cols, rows))
                continue;
            const Plane& plane_p = planes2[ny][nx];
            float new_cost = planeMatchCost(plane_p, x, y, WINDOW_SIZE, image_view);
            if (new_cost < old_cost) {
                old_plane = plane_p;
                old_cost = new_cost;
            }
        }
    }
}

void PatchMatch::viewPropagation(int x, int y, int image_view) {
    int sign = (image_view == 0) ? -1 : 1;  // -1 processing left, +1 processing right
    // current plane
    Plane view_plane;
    if (sign < 0) {
        view_plane = planes1[y][x];
    } else {
        view_plane = planes2[y][x];
    }

    // computing matching point in other view
    // reparameterized corresopndent plane in other view
    int mx, my;
    const Plane& new_plane = view_plane.viewTransform(x, y, sign, mx, my);
    if (!isInside(mx, my, 0, 0, cols, rows))
        return;
    // check the condition
    float& old_cost = costs[1 - image_view](my, mx);
    float new_cost = planeMatchCost(new_plane, mx, my, WINDOW_SIZE, 1 - image_view);
    if (new_cost < old_cost) {
        if (sign < 0) {
            planes2[my][mx] = new_plane;
        } else {
            planes1[my][mx] = new_plane;
        }
        old_cost = new_cost;
    }
}

void PatchMatch::preComputePixelsWeights(
    const cv::Mat3b& frame, cv::Mat& weights, int window_length) {
    int half = window_length / 2;
#pragma omp parallel for num_threads(8) collapse(2)
    for (int cx = 0; cx < frame.cols; cx++) {
        for (int cy = 0; cy < frame.rows; cy++) {
            for (int x = cx - half; x <= cx + half; x++) {
                for (int y = cy - half; y <= cy + half; y++) {
                    if (isInside(x, y, 0.0, 0.0, frame.cols, frame.rows)) {
                        weights.at<float>(cv::Vec4i{cy, cx, y - cy + half, x - cx + half}) =
                            weight(frame(cy, cx), frame(y, x), gamma_);
                    }
                }
            }
        }
    }
}

void PatchMatch::weightedMedianFilter(
    int cx, int cy, cv::Mat1f& disparity, const cv::Mat& weights, const cv::Mat1b& valid,
    int window_length, bool use_invalid) {
    int half = window_length / 2;
    float w_total = 0;
    float w = 0;
    std::vector<std::pair<float, float>> disps_window;
    for (int x = cx - half; x <= cx + half; x++) {
        for (int y = cy - half; y <= cy + half; y++) {
            if (isInside(x, y, 0.0, 0.0, cols, rows) && (use_invalid || valid(y, x))) {
                cv::Vec<int, 4> w_ids({cy, cx, y - cy + half, x - cx + half});
                w_total += weights.at<float>(w_ids);
                disps_window.push_back(std::make_pair(weights.at<float>(w_ids), disparity(y, x)));
            }
        }
    }
    std::sort(disps_window.begin(), disps_window.end());
    float med_weight = w_total / 2;
    for (std::size_t i = 0; i < disps_window.size(); i++) {
        /* code */
        w += disps_window[i].first;
        if (w >= med_weight) {
            if (i == 0) {
                disparity(cy, cx) = disps_window[i].second;
            } else {
                disparity(cy, cx) = (disps_window[i - 1].second + disps_window[i].second) / 2.0;
            }
        }
    }
}

// aggregated matchig cost of a plane for a pixel
float PatchMatch::planeMatchCost(
    const Plane& p, int cx, int cy, int window_length, int image_view) {
    int sign = -1 + 2 * image_view;
    float cost = 0.0;
    int half = window_length / 2;
    const cv::Mat3b& f1 = views[image_view];
    const cv::Mat3b& f2 = views[1 - image_view];
    const cv::Mat2f& g1 = grads[image_view];
    const cv::Mat2f& g2 = grads[1 - image_view];
    const cv::Mat& w1 = weights[image_view];
    for (int x = cx - half; x <= cx + half; x++) {
        for (int y = cy - half; y <= cy + half; y++) {
            if (!isInside(x, y, 0, 0, cols, rows))
                continue;
            // first compute q's disparity
            float dsp = disparity(x, y, p);
            if (dsp < 0.0 || dsp > MAX_DISPARITY) {
                cost += PLANE_PENALTY;
            } else {
                // find matching point in other view q' by subtracting the disparity from q's
                // x-coordinate
                float match = x + sign * dsp;
                int mathc_integer = static_cast<int>(match);
                float wm = 1 - (match - mathc_integer);
                if (mathc_integer > cols - 2)
                    mathc_integer = cols - 2;
                if (mathc_integer < 0)
                    mathc_integer = 0;

                // evaluating its color and gradient by linear interpolation
                cv::Vec3b color_value =
                    linearInterpolation(f2(y, mathc_integer), f2(y, mathc_integer + 1), wm);
                cv::Vec2b gradients_value =
                    linearInterpolation(g2(y, mathc_integer), g2(y, mathc_integer + 1), wm);
                float w = w1.at<float>(cv::Vec<int, 4>{cy, cx, y - cy + half, x - cx + half});
                cost += w * disSimilarity(f1(y, x), color_value, g1(y, x), gradients_value);
            }
        }
    }
    return cost;
}

void PatchMatch::fillInvalidPixels(int y, int x, Plane** planes, const cv::Mat1b& validity) {
    int x_left = x - 1;
    int x_right = x + 1;

    while (!validity(y, x_left) && x_left >= 0) {
        --x_left;
    }
    while (!validity(y, x_right) && x_left < cols) {
        ++x_right;
    }
    int bestplane_x = x;
    if (x_left >= 0 && x_right < cols) {
        float disp_l = disparity(x, y, planes[y][x_left]);
        float disp_r = disparity(x, y, planes[y][x_right]);
        bestplane_x = (disp_l < disp_r) ? x_left : x_right;
    } else if (x_left >= 0) {
        bestplane_x = x_left;
    } else if (x_right < cols) {
        bestplane_x = x_right;
    }
    planes[y][x] = planes[y][bestplane_x];
}

void PatchMatch::planesToDisparity(Plane** const planes, cv::Mat1f& disp) {
#pragma omp parallel for num_threads(8) collapse(2)
    for (int x = 0; x < cols; x++) {
        for (int y = 0; y < rows; y++) {
            disp(y, x) = disparity(x, y, planes[y][x]);
        }
    }
}

void PatchMatch::planeRefinement(
    int x, int y, int image_view, float max_delta_z, float max_delta_n, float end_delta_z) {
    int sign = (image_view == 0) ? -1 : 1;  // -1 processing left, +1 processing right
    float max_delta_z_local = max_delta_z;
    float max_delta_n_local = max_delta_n;
    float& old_costs = costs[image_view](y, x);
    if (sign < 0) {
        Plane& old_plane = planes1[y][x];
        while (max_delta_z_local >= end_delta_z) {
            std::random_device dev;
            std::mt19937 rng(dev());
            std::uniform_real_distribution<float> rand_z(-max_delta_z, max_delta_z);
            std::uniform_real_distribution<float> rand_n(-max_delta_n, max_delta_n);
            const auto& coeff = old_plane.getCoeff();
            float z = coeff[0] * x + coeff[1] * y + coeff[2];
            float delta_z = rand_z(rng);
            cv::Vec3f new_point(x, y, z + delta_z);
            const auto& n = old_plane.getNormal();
            cv::Vec3f delta_n(rand_n(rng), rand_n(rng), rand_n(rng));
            cv::Vec3f new_normal = n + delta_n;
            cv::normalize(new_normal, new_normal);

            // test new plane
            Plane new_plane(new_point, new_normal);
            float new_cost = planeMatchCost(new_plane, x, y, WINDOW_SIZE, image_view);
            if (new_cost < old_costs) {
                old_plane = new_plane;
                old_costs = new_cost;
            }
            max_delta_z_local /= 2;
            max_delta_n_local /= 2;
        }
    } else {
        Plane& old_plane = planes2[y][x];
        while (max_delta_z_local >= end_delta_z) {
            std::random_device dev;
            std::mt19937 rng(dev());
            std::uniform_real_distribution<float> rand_z(-max_delta_z, max_delta_z);
            std::uniform_real_distribution<float> rand_n(-max_delta_n, max_delta_n);
            const auto& coeff = old_plane.getCoeff();
            float z = coeff[0] * x + coeff[1] * y + coeff[2];
            float delta_z = rand_z(rng);
            cv::Vec3f new_point(x, y, z + delta_z);
            const auto& n = old_plane.getNormal();
            cv::Vec3f delta_n(rand_n(rng), rand_n(rng), rand_n(rng));
            cv::Vec3f new_normal = n + delta_n;
            cv::normalize(new_normal, new_normal);

            // test new plane
            Plane new_plane(new_point, new_normal);
            float new_cost = planeMatchCost(new_plane, x, y, WINDOW_SIZE, image_view);
            if (new_cost < old_costs) {
                old_plane = new_plane;
                old_costs = new_cost;
            }
            max_delta_z_local /= 2;
            max_delta_n_local /= 2;
        }
    }
}

void PatchMatch::init(const cv::Mat3b& img1, const cv::Mat3b& img2) {
    assert(img1.rows == img2.rows);
    assert(img1.cols == img2.cols);
    views[0] = img1;
    views[1] = img2;
    rows = img1.rows;
    cols = img1.cols;
    planes1 = new Plane*[rows];
    planes2 = new Plane*[rows];
    for (int i = 0; i < rows; i++) {
        planes1[i] = new Plane[cols];
        planes2[i] = new Plane[cols];
    }
    // std::cout << "Precomputing pixels weight" << std::endl;
    int weightmat_size[] = {rows, cols, WINDOW_SIZE, WINDOW_SIZE};
    weights[0] = cv::Mat(4, weightmat_size, CV_32F);
    weights[1] = cv::Mat(4, weightmat_size, CV_32F);
    preComputePixelsWeights(img1, weights[0], WINDOW_SIZE);
    preComputePixelsWeights(img2, weights[1], WINDOW_SIZE);
    // greyscale images gradient
    // std::cout << "Evaluating images gradient" << std::endl;
    grads[0] = cv::Mat2f(rows, cols);
    grads[1] = cv::Mat2f(rows, cols);
    computeGreyGradient(img1, grads[0]);
    computeGreyGradient(img2, grads[1]);

    // std::cout << "Precomputing random planes" << std::endl;
    initializeRandomPlanes(planes1, MAX_DISPARITY);
    initializeRandomPlanes(planes2, MAX_DISPARITY);

    // std::cout << "Evaluating initial planes cost" << std::endl;
    costs[0] = cv::Mat1f(rows, cols);
    costs[1] = cv::Mat1f(rows, cols);
#pragma omp parallel for num_threads(8) collapse(2)
    for (int y = 0; y < rows; y++) {
        for (int x = 0; x < cols; x++) {
            costs[0](y, x) = planeMatchCost(planes1[y][x], x, y, WINDOW_SIZE, 0);
        }
    }
#pragma omp parallel for num_threads(8) collapse(2)
    for (int y = 0; y < rows; y++) {
        for (int x = 0; x < cols; x++) {
            costs[1](y, x) = planeMatchCost(planes2[y][x], x, y, WINDOW_SIZE, 1);
        }
    }
    // left and right disparity maps
    disps[0] = cv::Mat1f(rows, cols);
    disps[1] = cv::Mat1f(rows, cols);
}

void PatchMatch::propagation(int iteration, bool reverse) {
    // std::cout << "Processing left and right views" << std::endl;
    for (int i = 0 + reverse; i < iteration + reverse; i++) {
        bool itertype = (i % 2 == 0);
        // std::cout << "Iteration " << i - reverse + 1 << "/" << iteration - reverse << std::endl;
        for (int work_view = 0; work_view < 2; work_view++) {
            if (itertype) {
                // even iteration star with left top pixel
                for (int y = 0; y < rows; y++) {
                    for (int x = 0; x < cols; x++) {
                        spatialPropagation(x, y, work_view, i);
                        viewPropagation(x, y, work_view);
                        planeRefinement(x, y, work_view, MAX_DISPARITY / 2, 1.0f, 1.0f);
                    }
                }
            } else {
                // odd iteration star with right bottom pixel
                for (int y = rows - 1; y >= 0; y--) {
                    for (int x = cols - 1; x >= 0; x--) {
                        spatialPropagation(x, y, work_view, i);
                        viewPropagation(x, y, work_view);
                        planeRefinement(x, y, work_view, MAX_DISPARITY / 2, 1.0f, 1.0f);
                    }
                }
            }
        }
    }
    planesToDisparity(planes1, disps[0]);
    planesToDisparity(planes2, disps[1]);
}

void PatchMatch::postProcess() {
    // checking pixels-plane disparity validity
    // std::cout << "Post Process" << std::endl;

    cv::Mat1b left_validity(rows, cols, (unsigned char)false);
    cv::Mat1b right_validity(rows, cols, (unsigned char)false);
    for (int y = 0; y < rows; y++) {
        for (int x = 0; x < cols; x++) {
            int x_match_right =
                std::max(0.0f, std::min(static_cast<float>(cols), x - disps[0](y, x)));
            left_validity(y, x) = (std::abs(disps[0](y, x) - disps[1](y, x_match_right)) <= 1);
            int x_match_left =
                std::max(0.0f, std::min(static_cast<float>(rows), x + disps[1](y, x)));
            right_validity(y, x) = (std::abs(disps[1](y, x) - disps[0](y, x_match_left)) <= 1);
        }
    }
    // cv::imwrite("l_inv.png", 255 * left_validity);
    // cv::imwrite("r_inv.png", 255 * right_validity);
// fill-in holes related to invalid pixels
#pragma omp parallel for num_threads(8) collapse(2)
    for (int y = 0; y < rows; y++) {
        for (int x = 0; x < cols; x++) {
            if (!left_validity(y, x))
                fillInvalidPixels(y, x, planes1, left_validity);

            if (!right_validity(y, x))
                fillInvalidPixels(y, x, planes2, right_validity);
        }
    }
    std::cout << "fillInvalidPixels Process" << std::endl;
    cv::Mat1b ld(rows, cols);
    cv::Mat1b rd(rows, cols);
    planesToDisparity(planes1, disps[0]);
    planesToDisparity(planes2, disps[1]);
    cv::normalize(disps[0], ld, 0, 255, cv::NORM_MINMAX);
    cv::normalize(disps[1], rd, 0, 255, cv::NORM_MINMAX);
    // cv::imwrite("ld2.png", ld);
    // cv::imwrite("rd2.png", rd);

    // applying weighted median filter to left and right view respectively
    for (int x = 0; x < cols; x++) {
        for (int y = 0; y < rows; y++) {
            weightedMedianFilter(x, y, disps[0], weights[0], left_validity, WINDOW_SIZE, false);
            weightedMedianFilter(x, y, disps[1], weights[1], right_validity, WINDOW_SIZE, false);
        }
    }
}

cv::Mat1f PatchMatch::getLeftDisparityMap() const {
    return disps[0];
}

cv::Mat1f PatchMatch::getRightDisparityMap() const {
    return disps[1];
}

}  // namespace stereo