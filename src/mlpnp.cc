#include "../include/mlpnp.hpp"

#include "../include/se3.hpp"
using namespace std;

MlPnPsolver::MlPnPsolver(
    const points_t& objectpoints, const pixels_t& imagepoints, Eigen::Matrix3d& rotation,
    Eigen::Vector3d& t)
    : inliers_num(0),
      miterations(0),
      mnbestinliers(0),
      N(0),
      objectpoints_(objectpoints),
      imagepoints_(imagepoints),
      pixel_noise_u(0.4),
      pixel_noise_v(0.4) {
    N = imagepoints.size();  // number of correspondences
    allindices_vec.reserve(N);
    bearingvec.reserve(N);
    covs.reserve(N);
    Eigen::Matrix3d K;
    // clang-format off
        K << 460, 0, 255,
                0, 460, 255,
                0, 0, 1;
    // clang-format on
    Eigen::DiagonalMatrix<double, 3> cov_xx(
        pixel_noise_u * pixel_noise_u, pixel_noise_v * pixel_noise_v, 0);
    Eigen::Matrix3d cov_proj = K.inverse() * cov_xx * K.transpose();
    for (unsigned int i = 0; i < N; i++) {
        Eigen::Vector3d unproject_points = unproject(imagepoints[i]);
        // Bearing vector should be normalized
        Bearingvector v = unproject_points.normalized();
        bearingvec.push_back(v);
        allindices_vec.push_back(i);
        Eigen::Matrix3d J =
            (Eigen::Matrix3d::Identity() - v * v.transpose()) / unproject_points.norm();
        Eigen::Matrix3d cov_vv = J * cov_proj * J.transpose();
        covs.push_back(cov_vv);
    }

    setRansacParameters();
    elapsed_seconds = 0.0;
}

MlPnPsolver::~MlPnPsolver() = default;

void MlPnPsolver::setRansacParameters(
    double probabity, std::size_t mininliers, std::size_t maxiterations, std::size_t minset,
    float epsilon, float th2) {
    ransacprob = probabity;
    ransacmininliers = mininliers;
    ransacmaxiters = maxiterations;
    ransacminset = minset;
    ransacepsilon = epsilon;

    inliers_vec.resize(N);
    // Adjust Parameters according to number of correspondences
    mininliers = static_cast<std::size_t>(N * ransacepsilon);
    if (mininliers < std::min(ransacmininliers, minset)) {
        mininliers = std::min(ransacmininliers, minset);
    }
    ransacmininliers = mininliers;
    if (ransacepsilon < static_cast<float>(ransacmininliers / N))
        ransacepsilon = static_cast<float>(ransacmininliers / N);
    // Set RANSAC iterations according to probability, epsilon, and max iterations
    std::size_t iterations;
    if (ransacmininliers == N)
        iterations = 1;
    else
        iterations = ceil(log(1.0 - ransacprob) / log(1.0 - pow(ransacepsilon, 3)));

    ransacmaxiters = std::max(static_cast<std::size_t>(1), std::min(iterations, ransacmaxiters));
}

// RANSAC methods
Eigen::Matrix4d MlPnPsolver::iterate(
    std::size_t iterations, std::vector<int8_t>& binliers, int& inliers) {
    binliers.clear();
    inliers = 0;
    if (N < ransacmininliers) {
        std::cerr << N << " " << ransacmininliers << std::endl;
        return Eigen::Matrix4d::Identity();
    }
    std::vector<size_t> availableindices;
    while (miterations < ransacmaxiters || miterations < iterations) {
        miterations++;
        availableindices = allindices_vec;
        points_t p3Ds(ransacminset);
        std::vector<std::size_t> indexes(ransacminset);
        std::vector<Bearingvector, Eigen::aligned_allocator<Bearingvector>> bearingvecRansac(
            ransacminset);
        // get min set of points
        // int idx[6] = {31, 29, 26, 35, 34, 6};
        // cov3_t covs(1);  // By the moment, we are using MLPnP without covariance info
        cov3_t covs_ransac(ransacminset);
        for (size_t i = 0; i < ransacminset; i++) {
            int rand_index = rng_generator(0, availableindices.size() - 1);
            int idx = availableindices[rand_index];

            bearingvecRansac[i] = bearingvec[idx];
            p3Ds[i] = objectpoints_[idx];
            covs_ransac[i] = covs[idx];
            indexes[i] = i;
            availableindices[rand_index] = availableindices.back();
            availableindices.pop_back();
            // std::cout << "random index " << idx << std::endl;
        }
        Eigen::Matrix4d result = Eigen::Matrix4d::Identity();

        gettimeofday(&start, nullptr);
        computePose(bearingvecRansac, p3Ds, covs_ransac, indexes, result);
        gettimeofday(&end, nullptr);
        elapsed_seconds = (end.tv_sec - start.tv_sec) * 1e3;
        elapsed_seconds += (end.tv_usec - start.tv_usec) * 1e-3;
        std::cout << "compute pose time: " << elapsed_seconds << std::endl;
        // save result
        mRi = result.block<3, 3>(0, 0);
        mti = result.block<3, 1>(0, 3);
        checkInliers();
        // std::cout << "inliners num: " << inliers_num << std::endl;
        if (inliers_num >= ransacmininliers) {
            if (inliers_num > mnbestinliers) {
                bestinliers_vec = inliers_vec;
                mnbestinliers = inliers_num;
            }
            if (refine()) {
                inliers = refinedinliers;
                binliers = std::vector<int8_t>(N, 0);
                for (size_t i = 0; i < N; i++) {
                    if (refinedinliers_vec[i]) {
                        binliers[i] = 1;
                    }
                }
                return mTcwi;
            }
        }
    }
    return Eigen::Matrix4d::Identity();
}

void MlPnPsolver::checkInliers() {
    inliers_num = 0;
    for (std::size_t i = 0; i < N; i++) {
        Eigen::Vector3d p = objectpoints_[i];
        Eigen::Vector3d p_cam = mRi * p + mti;
        p_cam = p_cam / p_cam(2, 0);
        Eigen::Vector2d dist = project(p_cam) - imagepoints_[i];
        if (dist.norm() < 0.8 * 5.991) {
            inliers_vec[i] = 1;
            inliers_num++;
        } else {
            inliers_vec[i] = 0;
            std::cout << "outlier distance " << dist.norm() << std::endl;
            std::cout << "T\n" << mRi << std::endl;
            std::cout << mti << std::endl;
        }
    }
};

// non-linear refinement
bool MlPnPsolver::refine() {
    std::vector<std::size_t> indices;
    indices.reserve(bestinliers_vec.size());
    for (std::size_t i = 0; i < bestinliers_vec.size(); i++) {
        if (bestinliers_vec[i])
            indices.push_back(i);
    }

    // Bearing vector and 3D points used for this ransac iteration
    std::vector<Bearingvector, Eigen::aligned_allocator<Bearingvector>> bearing_vec;
    points_t points_3d;
    std::vector<std::size_t> indexes;
    // By the moment, we are using MLPnP without covariance info
    cov3_t covs_matrix;
    for (std::size_t i = 0; i < indices.size(); i++) {
        auto idx = indices[i];
        bearing_vec.push_back(bearingvec[idx]);
        points_3d.push_back(objectpoints_[idx]);
        covs_matrix.push_back(covs[idx]);
        indexes.push_back(i);
    }

    Eigen::Matrix4d result = Eigen::Matrix4d::Identity();
    computePose(bearing_vec, points_3d, covs, indexes, result);
    // save result
    // std::cout << "result:\n" << result << std::endl;
    mRi = result.block<3, 3>(0, 0);
    mti = result.block<3, 1>(0, 3);
    checkInliers();
    refinedinliers = inliers_num;
    refinedinliers_vec = inliers_vec;
    if (refinedinliers > ransacmininliers) {
        mTcwi = result;
        return true;
    }
    return false;
}

// MLPnP method
void MlPnPsolver::computePose(
    const std::vector<Bearingvector, Eigen::aligned_allocator<Bearingvector>>& f,
    const points_t& points3d, const cov3_t& cov_mats, const std::vector<size_t>& indices,
    Eigen::Matrix4d& result) {
    std::size_t numberCorrespondences = indices.size();
    assert(numberCorrespondences > 5);

    bool planar = false;

    // compute the nullspace of all bearing vectors
    std::vector<Eigen::Matrix<double, 3, 2>> nullspaces(numberCorrespondences);
    Eigen::MatrixXd M(3, numberCorrespondences);
    points_t point3v(numberCorrespondences);
    for (std::size_t i = 0; i < numberCorrespondences; i++) {
        Bearingvector v = f[indices[i]];
        M.col(i) = points3d[indices[i]];
        // SVD of bearing vector
        Eigen::JacobiSVD<Eigen::MatrixXd, Eigen::HouseholderQRPreconditioner> svd_v(
            v.transpose(), Eigen::ComputeFullV);
        nullspaces[i] = svd_v.matrixV().block<3, 2>(0, 1);
        point3v[i] = points3d[indices[i]];
    }
    // 1. test if we have a planar scene (3.5 Planar case)
    Eigen::Matrix3d S = M * M.transpose().eval();
    Eigen::FullPivHouseholderQR<Eigen::Matrix3d> ranktest(S);
    Eigen::Matrix3d R_s = Eigen::Matrix3d::Identity();

    if (ranktest.rank() == 2) {
        planar = true;
        // self adjoint is faster and more accurate than general eigen solvers
        // also has closed form solution for 3x3 self-adjoint matrices
        // in addition this solver sorts the eigenvalues in increasing order
        Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> eigen_solver(S);
        R_s = eigen_solver.eigenvectors().real();
        R_s.transposeInPlace();
        for (size_t i = 0; i < numberCorrespondences; i++) {
            M.col(i) = R_s * M.col(i);
        }
    }
    // 2. stochastic model
    Eigen::SparseMatrix<double> P(2 * numberCorrespondences, 2 * numberCorrespondences);
    bool use_cov{false};
    P.setIdentity();
    // if we do have covariance information
    // -> fill covariance matrix
    if (cov_mats.size() == numberCorrespondences) {
        use_cov = true;
        int l = 0;
        for (size_t i = 0; i < numberCorrespondences; i++) {
            Eigen::Matrix2d temp =
                nullspaces[i].transpose().eval() * cov_mats[i] * nullspaces[i];  // equ 9
            temp = temp.inverse().eval();
            P.coeffRef(l, l) = temp(0, 0);
            P.coeffRef(l, l + 1) = temp(0, 1);
            P.coeffRef(l + 1, l) = temp(1, 0);
            P.coeffRef(l + 1, l + 1) = temp(1, 1);
            l += 2;
        }
    }
    // 3. fill the design matrix A
    Eigen::MatrixXd A;
    if (planar) {
        A = Eigen::MatrixXd(2 * numberCorrespondences, 9);
    } else {
        A = Eigen::MatrixXd(2 * numberCorrespondences, 12);
    }
    A.setZero();

    // fill design matrix
    if (planar) {
        for (size_t i = 0; i < numberCorrespondences; i++) {
            const Eigen::Vector3d& pt3_current = M.col(i);
            const Eigen::Matrix<double, 3, 2>& nullspace = nullspaces[i];
            Eigen::Matrix<double, 2, 9> A_temp;
            double r1py = nullspace(0, 0) * pt3_current[1];
            double s1py = nullspace(0, 1) * pt3_current[1];
            double r1pz = nullspace(0, 0) * pt3_current[2];
            double s1pz = nullspace(0, 1) * pt3_current[2];
            double r2py = nullspace(1, 0) * pt3_current[1];
            double s2py = nullspace(1, 1) * pt3_current[1];
            double r2pz = nullspace(1, 0) * pt3_current[2];
            double s2pz = nullspace(1, 1) * pt3_current[2];
            double r3py = nullspace(2, 0) * pt3_current[1];
            double s3py = nullspace(2, 1) * pt3_current[1];
            double r3pz = nullspace(2, 0) * pt3_current[2];
            double s3pz = nullspace(2, 1) * pt3_current[2];
            // clang-format off
            A_temp << r1py, r1pz, r2py, r2pz, r3py, r3pz, nullspace(0, 0), nullspace(1, 0), nullspace(2, 0),
                                   s1py, s1pz, s2py, s2pz, s3py, s3pz, nullspace(0, 1), nullspace(1, 1), nullspace(2, 1);
            // clang-format on
            A.block<2, 9>(2 * i, 0) = A_temp;
        }
    } else {
        for (size_t i = 0; i < numberCorrespondences; i++) {
            const Eigen::Vector3d& pt3_current = M.col(i);
            const Eigen::Matrix<double, 3, 2>& nullspace = nullspaces[i];
            Eigen::Matrix<double, 2, 12> A_temp;  // 公式 （11）
            double r1px = nullspace(0, 0) * pt3_current[0];
            double s1px = nullspace(0, 1) * pt3_current[0];
            double r1py = nullspace(0, 0) * pt3_current[1];
            double s1py = nullspace(0, 1) * pt3_current[1];
            double r1pz = nullspace(0, 0) * pt3_current[2];
            double s1pz = nullspace(0, 1) * pt3_current[2];
            double r2px = nullspace(1, 0) * pt3_current[0];
            double s2px = nullspace(1, 1) * pt3_current[0];
            double r2py = nullspace(1, 0) * pt3_current[1];
            double s2py = nullspace(1, 1) * pt3_current[1];
            double r2pz = nullspace(1, 0) * pt3_current[2];
            double s2pz = nullspace(1, 1) * pt3_current[2];
            double r3px = nullspace(2, 0) * pt3_current[0];
            double s3px = nullspace(2, 1) * pt3_current[0];
            double r3py = nullspace(2, 0) * pt3_current[1];
            double s3py = nullspace(2, 1) * pt3_current[1];
            double r3pz = nullspace(2, 0) * pt3_current[2];
            double s3pz = nullspace(2, 1) * pt3_current[2];
            // clang-format off
            A_temp << r1px, r1py, r1pz, r2px, r2py, r2pz, r3px, r3py, r3pz, nullspace(0, 0), nullspace(1, 0), nullspace(2, 0),
                                    s1px, s1py, s1pz, s2px, s2py, s2pz, s3px, s3py, s3pz, nullspace(0, 1), nullspace(1, 1), nullspace(2, 1);
            // clang-format on
            A.block<2, 12>(2 * i, 0) = A_temp;
        }
    }
    // 4. solve least squares
    Eigen::MatrixXd N_matrix;
    if (use_cov)
        N_matrix = A.transpose().eval() * P * A;  // equ (14)
    else
        N_matrix = A.transpose().eval() * A;
    Eigen::JacobiSVD<Eigen::MatrixXd> svd_A(N_matrix, Eigen::ComputeFullV);
    Eigen::MatrixXd result1 = svd_A.matrixV().col(A.cols() - 1);
    // now we treat the results differently,
    // depending on the scene (planar or not)
    Eigen::Matrix3d Rout = Eigen::Matrix3d::Identity();
    Eigen::Vector3d tout = Eigen::Vector3d::Zero();
    if (planar) {
        tout << result1(6, 0), result1(7, 0), result1(8, 0);
        Eigen::Matrix3d R_hat;
        // clang-format off
        R_hat << 0.0, result1(0, 0), result1(1, 0),
                0.0, result1(2, 0), result1(3, 0),
                0.0, result1(4, 0), result1(5, 0);
        // clang-format on
        R_hat.col(0) = R_hat.col(1).cross(R_hat.col(2));
        R_hat.transposeInPlace();
        double scale = 1.0 / std::sqrt(std::abs(R_hat.col(1).norm() * R_hat.col(2).norm()));
        // find best rotation matrix in frobenius sense
        Eigen::JacobiSVD<Eigen::MatrixXd> svd_R_frob(
            R_hat, Eigen::ComputeFullU | Eigen::ComputeFullV);
        R_hat = svd_R_frob.matrixU() * svd_R_frob.matrixV().transpose();
        // test if we found a good rotation matrix
        if (R_hat.determinant() < 0)
            R_hat *= -1;
        // rotate this matrix back using the eigen frame
        R_hat = R_s.transpose().eval() * R_hat;
        tout = scale * tout;
        R_hat.transposeInPlace();
        R_hat *= -1;
        if (R_hat.determinant() < 0)
            R_hat.col(2) *= -1.0;
        Eigen::Matrix3d R1 = R_hat;
        Eigen::Matrix3d R2 = R_hat;
        R2.col(0) *= -1;
        R2.col(1) *= -1;
        std::vector<Eigen::Matrix4d, Eigen::aligned_allocator<Eigen::Matrix4d>> Ts(
            4, Eigen::Matrix4d::Identity());
        Ts[0].block<3, 3>(0, 0) = R1;
        Ts[0].block<3, 1>(0, 3) = tout;
        Ts[1].block<3, 3>(0, 0) = R1;
        Ts[1].block<3, 1>(0, 3) = -tout;
        Ts[2].block<3, 3>(0, 0) = R2;
        Ts[2].block<3, 1>(0, 3) = tout;
        Ts[3].block<3, 3>(0, 0) = R2;
        Ts[3].block<3, 1>(0, 3) = -tout;
        std::vector<double> diff1(4);
        for (size_t i = 0; i < 4; i++) {
            Eigen::Vector3d resultPt;
            double testres = 0.0;
            for (size_t j = 0; j < 6; j++) {
                resultPt = Ts[i].block<3, 3>(0, 0) * point3v[j] + Ts[i].block<3, 1>(0, 3);
                resultPt.normalize();
                testres += (1.0 - resultPt.transpose() * f[indices[j]]);
            }
            diff1[i] = testres;
        }
        auto minres_iter = std::min_element(std::begin(diff1), std::end(diff1));
        int index = std::distance(std::begin(diff1), minres_iter);
        Rout = Ts[index].block<3, 3>(0, 0);
        tout = Ts[index].block<3, 1>(0, 3);
    } else {  // non-planar
        // get the scale
        Eigen::Matrix3d R_hat;
        // clang-format off
        R_hat << result1(0, 0), result1(3, 0), result1(6, 0),
                result1(1, 0), result1(4, 0), result1(7, 0),
                result1(2, 0), result1(5, 0), result1(8, 0);
        tout << result1(9, 0), result1(10, 0), result1(11, 0);
        // clang-format on
        double scale = 1.0 / std::pow(
                                 std::abs(R_hat.col(0).norm()) * std::abs(R_hat.col(1).norm()) *
                                     std::abs(R_hat.col(2).norm()),
                                 1.0 / 3.0);
        // find best rotation matrix in frobenius sense
        Eigen::JacobiSVD<Eigen::MatrixXd> svdR_hat(
            R_hat, Eigen::ComputeFullU | Eigen::ComputeFullV);
        Rout = svdR_hat.matrixU() * svdR_hat.matrixV().transpose();
        // test if we have a good rotation matrix
        if (Rout.determinant() < 0)
            Rout *= -1.0;
        tout = Rout * (scale * tout);
        // find correct direction in terms of reprojection error, just take the first 6
        // correspondences
        double error[2];
        std::vector<Eigen::Matrix4d, Eigen::aligned_allocator<Eigen::Matrix4d>> Ts(
            2, Eigen::Matrix4d::Identity());
        for (size_t s = 0; s < 2; s++) {
            error[s] = 0.0;
            Ts[s].block<3, 3>(0, 0) = Rout;
            if (s == 0)
                Ts[s].block<3, 1>(0, 3) = tout;
            else
                Ts[s].block<3, 1>(0, 3) = -tout;
            Ts[s] = Ts[s].inverse().eval();
            for (size_t p = 0; p < 6; p++) {
                Bearingvector v = Ts[s].block<3, 3>(0, 0) * point3v[p] + Ts[s].block<3, 1>(0, 3);
                v.normalize();
                error[s] += (1.0 - v.transpose() * f[indices[p]]);
            }
        }
        if (error[0] < error[1])
            tout = Ts[0].block<3, 1>(0, 3);
        else
            tout = Ts[1].block<3, 1>(0, 3);
        Rout = Ts[0].block<3, 3>(0, 0);
    }
    result.block<3, 3>(0, 0) = Rout;
    result.block<3, 1>(0, 3) = tout;
    // 5. gauss newton
    mlPnPGN(result, point3v, nullspaces, P, use_cov);
}

void MlPnPsolver::mlPnPGN(
    Eigen::Matrix4d& T, const points_t& pts,
    const std::vector<Eigen::Matrix<double, 3, 2>>& nullspaces,
    const Eigen::SparseMatrix<double>& cov, bool use_cov) {
    const int numobservations = pts.size();
    const int numunknowns = 6;
    // redundancy
    int redundanz = 2 * numobservations - numunknowns;

    int cnt = 0;
    bool stop = false;
    constexpr int maxiternum = 5;
    double epsP = 1e-5;
    Eigen::VectorXd r(2 * numobservations);
    Eigen::MatrixXd jacobian(2 * numobservations, numunknowns);
    Eigen::MatrixXd jacTcov;
    Eigen::MatrixXd A;
    Eigen::Matrix<double, 6, 1> g, dx;
    while (cnt < maxiternum && !stop) {
        mlPnPResidualJacs(T, pts, nullspaces, r, jacobian, true);
        if (use_cov)
            jacTcov = jacobian.transpose() * cov;
        else
            jacTcov = jacobian.transpose();
        A = jacTcov * jacobian;

        g = jacTcov * r;

        // solve
        Eigen::LDLT<Eigen::MatrixXd> chol(A);
        dx = chol.solve(g);
        // this is to prevent the solution from falling into a wrong minimum
        // if the linear estimate is spurious
        if (dx.array().abs().maxCoeff() > 5.0 || dx.array().abs().minCoeff() > 1.0)
            break;
        // observation update
        Eigen::MatrixXd dl = jacobian * dx;
        if (dl.array().abs().maxCoeff() < epsP) {
            stop = true;
            T = se3exp(dx) * T;
            break;
        } else
            T = se3exp(dx) * T;
        ++cnt;
    }  // while
}

void MlPnPsolver::mlPnPResidualJacs(
    const Eigen::Matrix4d& T, const points_t& pts,
    const std::vector<Eigen::Matrix<double, 3, 2>>& nullspaces, Eigen::VectorXd& r,
    Eigen::MatrixXd& fjac, bool getjacs) {
    Eigen::Matrix<double, 2, 6> J;
    for (size_t i = 0; i < pts.size(); i++) {
        Eigen::Vector3d res = T.block<3, 3>(0, 0) * pts[i] + T.block<3, 1>(0, 3);
        if (getjacs) {
            mlPnPJacs(pts[i], nullspaces[i], res, J);
            fjac.block<2, 6>(2 * i, 0) = J;
        }
        res.normalize();
        r[2 * i] = nullspaces[i].col(0).transpose() * res;
        r[2 * i + 1] = nullspaces[i].col(1).transpose() * res;
    }
}

void MlPnPsolver::mlPnPJacs(
    const Eigen::Vector3d& pt, const Eigen::Matrix<double, 3, 2>& nullspace,
    const Eigen::Vector3d& vec, Eigen::Matrix<double, 2, 6>& jacs) {
    Eigen::Matrix3d temp_skew = Skew(vec);
    Eigen::Matrix<double, 3, 6> J;
    J.block<3, 3>(0, 0).setIdentity();
    J.block<3, 3>(0, 3) = -temp_skew;
    jacs = nullspace.transpose().eval() * J;
}