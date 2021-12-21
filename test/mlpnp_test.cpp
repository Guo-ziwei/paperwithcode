#include "../include/mlpnp.hpp"

#include <Eigen/Eigen>
#include <fstream>
#include <iostream>
#include <opencv2/calib3d.hpp>
#include <opencv2/core/eigen.hpp>
#include <string>

using namespace cv;

bool loadPointsEigen(
    std::string filename,
    std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>>& pointsvec,
    std::vector<Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d>>& pixelvec) {
    std::ifstream f;
    f.open(filename.c_str());
    if (!f.is_open()) {
        std::cerr << " can't open LoadFeatures file " << std::endl;
        return false;
    }
    Eigen::Vector3d uvz = Eigen::Vector3d(1, 1, 1);
    Eigen::Matrix3d K;
    // clang-format off
    K << 460 , 0, 255, 
        0, 460, 255,
        0, 0, 1;
    // clang-format on
    while (!f.eof()) {
        std::string s;
        std::getline(f, s);

        if (!s.empty()) {
            std::stringstream ss;
            ss << s;
            int i;
            Eigen::Vector2d pixel;
            Eigen::Vector3d points;

            ss >> points[0];
            ss >> points[1];
            ss >> points[2];
            ss >> i;
            ss >> uvz.x();
            ss >> uvz.y();
            uvz = K * uvz;
            pixel[0] = uvz.x();
            pixel[1] = uvz.y();
            pointsvec.push_back(points);
            pixelvec.push_back(pixel);
        }
    }
    return true;
}

int main(int argc, char const* argv[]) {
    std::string filename0(argv[1]);
    // std::string filename1 = "/home/guoziwei/git/vio_data_simulation/keyframe/all_points_1.txt";

    std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>> pointsvec_eigen;
    std::vector<Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d>> pixelvec_eigen;
    loadPointsEigen(filename0, pointsvec_eigen, pixelvec_eigen);

    Eigen::Vector3d t_w_c;
    Eigen::Matrix3d R_w_c;
    Eigen::Matrix4d T;
    MlPnPsolver mlpnp(pointsvec_eigen, pixelvec_eigen, R_w_c, t_w_c);
    std::vector<int8_t> inliersvec;
    int inliner_num = 0;
    T = mlpnp.iterate(10, inliersvec, inliner_num);
    Eigen::Matrix3d R_mlpnp = T.block<3, 3>(0, 0).transpose().eval();
    Eigen::Vector3d t_mlpnp = -R_mlpnp * T.block<3, 1>(0, 3);
    std::cout << Eigen::Quaterniond(R_mlpnp).coeffs() << std::endl;
    std::cout << t_mlpnp << std::endl;
    return 0;
}
