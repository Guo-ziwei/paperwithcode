#pragma once
#include <Eigen/Dense>
#include <Eigen/Eigen>

template <typename Derived>
inline Eigen::Matrix<typename Derived::Scalar, 3, 3> Skew(const Eigen::MatrixBase<Derived>& vec) {
    EIGEN_STATIC_ASSERT_VECTOR_SPECIFIC_SIZE(Derived, 3);
    // clang-format off
    return (Eigen::Matrix<typename Derived::Scalar, 3, 3>() << 0, -vec[2], vec[1],
                                                        vec[2], 0, -vec[0], 
                                                        -vec[1], vec[0], 0).finished();
    // clang-format on
}

template <typename Derived>
inline Eigen::Matrix<typename Derived::Scalar, 4, 4> se3exp(const Eigen::MatrixBase<Derived>& vec) {
    EIGEN_STATIC_ASSERT_VECTOR_SPECIFIC_SIZE(Derived, 6);
    const auto&& rvec = vec.template block<3, 1>(0, 0);
    const auto&& tvec = vec.template block<3, 1>(3, 0);
    const typename Derived::Scalar theta = rvec.norm();
    Eigen::Matrix<typename Derived::Scalar, 3, 1> u = rvec / theta;
    const auto&& u_skew = Skew(u);
    Eigen::Matrix<typename Derived::Scalar, 3, 3> C =
        Eigen::Matrix<typename Derived::Scalar, 3, 3>::Identity() + sin(theta) * u_skew +
        (1 - cos(theta)) * (u * u.transpose() - Eigen::Matrix<typename Derived::Scalar, 3, 3>::Identity());
    Eigen::Matrix<typename Derived::Scalar, 3, 3> J =
        sin(theta) / theta * Eigen::Matrix<typename Derived::Scalar, 3, 3>::Identity() +
        (1 - sin(theta) / theta) * (u * u.transpose()) + (1 - cos(theta)) / theta * u_skew;
    Eigen::Matrix<typename Derived::Scalar, 3, 1> r = J * tvec;
    Eigen::Matrix<typename Derived::Scalar, 4, 4> res = Eigen::Matrix<typename Derived::Scalar, 4, 4>::Identity();
    res.template block<3, 3>(0, 0) = C;
    res.template block<3, 1>(0, 3) = r;
    return res;
}