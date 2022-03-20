//
// Created by atway on 2022/3/20.
//

#ifndef ICP_PLICP_H
#define ICP_PLICP_H


#include <iostream>
#include <pcl/io/pcd_io.h>
#include <pcl/io/ascii_io.h>
#include <pcl/point_types.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/common/common.h>
#include <pcl/registration/registration.h>
#include <pcl/features/normal_3d.h>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <sophus/se3.hpp>
#include <numeric>
using namespace std;
using namespace Eigen;
/**
  1. 计算目标点云的法向量
  2. 查找最邻近点
  3. 根据距离对邻近点进行剔除,法相关系进行剔除
  4. 先小量的方式，解析解进行求解
  5, 迭代更新， 直接到变化量很小或者距离很小或者达到最大迭代次数停止
*/
class PLICP
{
public:
    typedef pcl::PointXYZ PointT;
    typedef pcl::PointCloud<PointT> PointCloudT;

    PLICP();

    void setInputSource(PointCloudT::Ptr& source);

    void setInputTarget(PointCloudT::Ptr& target);

    void align(PointCloudT& output);

    float computeRT(std::vector<PointT>& source, std::vector<PointT>& target, pcl::PointCloud <pcl::Normal>::Ptr& normals, Eigen::Matrix4f& T);

    void setMaximumIterations(int iterations);

    Eigen::Matrix4f getFinalTransformation();

    float getFitnessScore();

    void computeNormal(PointCloudT::Ptr& input, pcl::PointCloud <pcl::Normal>::Ptr& normals);

    void searchKNN(PointCloudT& input, pcl::PointCloud <pcl::Normal>::Ptr& normals, int k, std::vector<PointT>& matchInput, std::vector<PointT>& matchTarget);


private:
    float getLoss(std::vector<PointT>& source, std::vector<PointT>& target, Eigen::Matrix4f T);
    float computeProjectDist(PointT& p1, PointT& p2, pcl::Normal& n);
private:
    PointCloudT::Ptr m_source;
    PointCloudT::Ptr m_target;
    //最大迭代次数
    int m_max_iters;
    //输出最后的矩阵
    Eigen::Matrix4f m_T;

    pcl::KdTreeFLANN<PointT> m_kdtree;
};



#endif //ICP_PLICP_H
