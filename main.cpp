#include <iostream>
#include "ICP.h"
#include "PLICP.h"


void icp(pcl::PointCloud<ICP::PointT>::Ptr input, pcl::PointCloud<ICP::PointT>::Ptr target) {
    pcl::PointCloud<ICP::PointT>::Ptr output_cloud(new pcl::PointCloud<ICP::PointT>);
    ICP icp;
    icp.setInputSource(input);

    icp.setInputTarget(target);

    icp.setMaximumIterations(1000);

    icp.align(*output_cloud);

    Eigen::Matrix4f T =  icp.getFinalTransformation();

    //
    std::cout <<"ICP transformation\r\n" << T << endl;

    // 输出点云数据

    pcl::io::savePCDFile("../data/demo_icp.pcd", *output_cloud);
    std::cerr << "Saved " << output_cloud->size() << " data points to data/demo_icp.pcd." << std::endl;

}


void plicp(pcl::PointCloud<ICP::PointT>::Ptr input, pcl::PointCloud<ICP::PointT>::Ptr target){

    pcl::PointCloud<PLICP::PointT>::Ptr output_cloud(new pcl::PointCloud<PLICP::PointT>);

    PLICP plicp;
    plicp.setInputSource(input);

    plicp.setInputTarget(target);

    plicp.setMaximumIterations(1000);

    plicp.align(*output_cloud);

    Eigen::Matrix4f T = plicp.getFinalTransformation();

    //
    std::cout << "ICP transformation \r\n" << T << endl;

    // 输出点云数据
    pcl::io::savePCDFile("../data/demo_plicp.pcd", *output_cloud);
    std::cerr << "Saved " << output_cloud->size() << " data points to data/demo_plicp.pcd." << std::endl;
}

int  main() {

    pcl::PointCloud<ICP::PointT>::Ptr input_cloud(new pcl::PointCloud<ICP::PointT>);
    if (pcl::io::loadPCDFile<pcl::PointXYZ>("../data/input.pcd", *input_cloud) < 0)
    {
        PCL_ERROR("\a源点云文件不存在！\n");
        return -1;
    }
    pcl::PointCloud<ICP::PointT>::Ptr target_cloud(new pcl::PointCloud<ICP::PointT>);
    if (pcl::io::loadPCDFile<pcl::PointXYZ>("../data/target.pcd", *target_cloud) < 0)
    {
        PCL_ERROR("\a目标点云文件不存在！\n");
        return -1;
    }
    cout << "*********************icp***************************" << endl;
    icp(input_cloud, target_cloud);
    cout << "*********************plicp***************************" << endl;
    plicp(input_cloud, target_cloud);

}

