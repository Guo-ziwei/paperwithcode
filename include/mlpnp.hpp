#pragma once
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <cmath>
#include <iostream>
#include <random>
#include <sys/time.h>

class MlPnPsolver {
  public:
    using Bearingvector = Eigen::Vector3d;
    using points_t = std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>>;
    using pixels_t = std::vector<Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d>>;
    using cov3_t = std::vector<Eigen::Matrix3d, Eigen::aligned_allocator<Eigen::Matrix3d>>;
    MlPnPsolver(
        const points_t& objectpoints, const pixels_t& imagepoints, Eigen::Matrix3d& rotation,
        Eigen::Vector3d& t);
    ~MlPnPsolver();
    void setRansacParameters(
        double probabity = 0.99, std::size_t mininliers = 8, std::size_t maxiterations = 300,
        std::size_t minset = 10, float epsilon = 0.4, float th2 = 5.991);

    // RANSAC methods
    Eigen::Matrix4d iterate(std::size_t iterations, std::vector<int8_t>& binliers, int& inliers);

  private:
    void checkInliers();

    bool refine();
    /*
     * Computes the camera pose given 3D points coordinates (in the camera reference
     * system), the camera rays and (optionally) the covariance matrix of those camera rays.
     * Result is stored in solution
     */
    void computePose(
        const std::vector<Bearingvector, Eigen::aligned_allocator<Bearingvector>>& f,
        const points_t& points3d, const cov3_t& cov_mats, const std::vector<std::size_t>& indices,
        Eigen::Matrix4d& result);

    void mlPnPGN(
        Eigen::Matrix4d& T, const points_t& pts,
        const std::vector<Eigen::Matrix<double, 3, 2>>& nullspaces,
        const Eigen::SparseMatrix<double>& cov, bool use_cov);

    void mlPnPResidualJacs(
        const Eigen::Matrix4d& T, const points_t& pts,
        const std::vector<Eigen::Matrix<double, 3, 2>>& nullspaces, Eigen::VectorXd& r,
        Eigen::MatrixXd& fjac, bool getjacs);

    void mlPnPJacs(
        const Eigen::Vector3d& pt, const Eigen::Matrix<double, 3, 2>& nullspace,
        const Eigen::Vector3d& vec, Eigen::Matrix<double, 2, 6>& jacs);

    Eigen::Vector3d unproject(const Eigen::Vector2d& pixel) {
        Eigen::Matrix3d K;
        // clang-format off
        K << 460, 0, 255,
                0, 460, 255,
                0, 0, 1;
        // clang-format on
        return K.inverse() * Eigen::Vector3d(pixel.x(), pixel.y(), 1);
    }

    Eigen::Vector2d project(const Eigen::Vector3d& ptcam) {
        Eigen::Matrix3d K;
        // clang-format off
        K << 460, 0, 255,
                0, 460, 255,
                0, 0, 1;
        // clang-format on
        Eigen::Vector3d pixel = K * ptcam;
        return Eigen::Vector2d(pixel[0], pixel[1]);
    }

    // Current Estimation
    Eigen::Matrix3d mRi;
    Eigen::Vector3d mti;
    Eigen::Matrix4d mTcwi;
    std::vector<int8_t> inliers_vec;
    std::size_t inliers_num;
    // Current Ransac State
    std::size_t miterations;
    std::vector<int8_t> bestinliers_vec;

    std::vector<double> mvMaxError;
    std::size_t mnbestinliers;
    // Number of Correspondences
    std::size_t N;

    // Indices for random selection [0 .. N-1]
    std::vector<std::size_t> allindices_vec;

    // RANSAC probability
    double ransacprob;

    // RANSAC min inliers
    std::size_t ransacmininliers;

    // RANSAC max iterations
    std::size_t ransacmaxiters;

    // RANSAC Minimun Set used at each iteration
    std::size_t ransacminset;

    // RANSAC expected inliers/total ratio
    float ransacepsilon;

    // RANSAC Threshold inlier/outlier. Max error e = dist(P1,T_12*P2)^2
    float mRansacTh;

    // Refined
    std::vector<int8_t> refinedinliers_vec;
    std::size_t refinedinliers;

    std::vector<Bearingvector, Eigen::aligned_allocator<Bearingvector>> bearingvec;
    cov3_t covs;
    const points_t& objectpoints_;

    const pixels_t& imagepoints_;

    struct timeval start, end;

    double elapsed_seconds;

    double pixel_noise_u, pixel_noise_v;
};

inline int rng_generator(int range_begin, int range_end) {
    std::random_device dev;
    std::mt19937 rng(dev());
    std::uniform_int_distribution<std::mt19937::result_type> dist6(
        range_begin, range_end);  // distribution in range [0, 200]
    return dist6(rng);
}