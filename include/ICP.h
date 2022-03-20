//
// Created by atway on 2022/3/19.
//

#ifndef ICP_ICP_H
#define ICP_ICP_H

#include <iostream>
#include <pcl/io/pcd_io.h>
#include <pcl/io/ascii_io.h>
#include <pcl/point_types.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/common/common.h>
#include <pcl/registration/registration.h>
#include <pcl/features/normal_3d.h>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Eigen/Dense>
#include <sophus/se3.hpp>
#include <algorithm>
#include <numeric>
using namespace std;
using namespace Eigen;
/**
  1. 查找最邻近点
  2. 根据距离对邻近点进行剔除
  3. 计算svd 分解 R， t
  4, 迭代更新， 直接到变化量很小或者距离很小或者达到最大迭代次数停止
*/


class ICP
{
public:
    typedef pcl::PointXYZ PointT;
    typedef pcl::PointCloud<PointT> PointCloudT;

    ICP();

    void setInputSource(PointCloudT::Ptr& source);

    void setInputTarget(PointCloudT::Ptr& target);

    void align(PointCloudT& output);

    float computeRT(std::vector<PointT>& source, std::vector<PointT>& target, Eigen::Matrix4f& T);

    void setMaximumIterations(int iterations);

    Eigen::Matrix4f getFinalTransformation();

    float getFitnessScore();

    void searchKNN(PointCloudT& input, int k, std::vector<PointT>& matchInput, std::vector<PointT>& matchTarget);
private:
    float getLoss(std::vector<PointT>& source, std::vector<PointT>& target, Eigen::Matrix4f T);
private:
    PointCloudT::Ptr m_source;
    PointCloudT::Ptr m_target;
    //最大迭代次数
    int m_max_iters;
    //输出最后的矩阵
    Eigen::Matrix4f m_T;

    pcl::KdTreeFLANN<PointT> m_kdtree;
};



#endif //ICP_ICP_H
