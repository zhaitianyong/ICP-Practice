//
// Created by atway on 2022/3/20.
//

#include "PLICP.h"
#include "PLICP.h"
#include <pcl/visualization/pcl_visualizer.h>
PLICP::PLICP() {

    m_max_iters = 20;
    m_T = Matrix4f::Identity();
}

void PLICP::setInputSource(PointCloudT::Ptr& source) {

    m_source = source;
}

void PLICP::setInputTarget(PointCloudT::Ptr& target) {
    m_target = target;
    m_kdtree.setInputCloud(m_target);
}

void PLICP::setMaximumIterations(int iterations)
{
    this->m_max_iters = iterations;
}

Eigen::Matrix4f PLICP::getFinalTransformation()
{
    return m_T;
}

float PLICP::getFitnessScore()
{
    return 0.0f;
}

void PLICP::computeNormal(PointCloudT::Ptr& input, pcl::PointCloud <pcl::Normal>::Ptr& normals)
{
    pcl::NormalEstimation<PointT, pcl::Normal> ne;									//创建法线估计对象ne
    pcl::search::Search<PointT>::Ptr tree(new pcl::search::KdTree<PointT>);	//设置搜索方法
    ne.setSearchMethod(tree);
    ne.setInputCloud(input);
    ne.setKSearch(30);
    ne.compute(*normals);
}


void PLICP::searchKNN(PointCloudT& input, pcl::PointCloud <pcl::Normal>::Ptr& normals, int k, std::vector<PointT>& matchInput, std::vector<PointT>& matchTarget)
{
    std::vector<pair<int,int>> ids;
    std::vector<float> distances;
    {
        std::vector<int> pointIdxNKNSearch(k);
        std::vector<float> pointNKNSquaredDistance(k);
        for (int i=0; i<input.size(); i++)
        {
            if (m_kdtree.nearestKSearch(input.points[i], k, pointIdxNKNSearch, pointNKNSquaredDistance) > 0) {
                int& index = pointIdxNKNSearch[0];
                ids.push_back(make_pair(i,index));
                distances.push_back(computeProjectDist(input.points[i], m_target->points[index], normals->points[index]));
            }
        }
    }
    //对查找的点进行过滤
    {
        //计算平均值和标准差
        float min_planarity = 0.01; //根据需要进行修改
        float sum = std::accumulate(distances.begin(), distances.end(), 0.0);
        float mean = sum / distances.size();
        float accum = 0.0;
        std::for_each(distances.begin(), distances.end(), [&](const float d) {
            accum += (d - mean) * (d - mean);
        });
        float stdev = sqrt(accum / (distances.size() - 1));

        for (auto& id : ids)
        {
            //曲率判断
            if (normals->points[id.second].curvature < min_planarity)
                continue;
            //距离判断
            if (abs(distances[id.first] - mean) > 3 * stdev)
                continue;
            matchInput.push_back(input.points[id.first]);
            matchTarget.push_back(m_target->points[id.second]);
        }

    }

}

float PLICP::getLoss(std::vector<PointT>& source, std::vector<PointT>& target, Eigen::Matrix4f T)
{
    return 0.0f;
}

float PLICP::computeProjectDist(PointT& p1, PointT& p2, pcl::Normal& n)
{
    Vector3f p1p2, normal;
    p1p2 << p2.x - p1.x , p2.y - p1.y , p2.z - p1.z;
    normal << n.normal_x, n.normal_y, n.normal_z;
    return sqrt(abs(p1p2.dot(normal)));
}

void PLICP::align(PointCloudT& output) {

    pcl::copyPointCloud(*m_source, output);

    //1.计算目标点云的法向量和曲率
    pcl::PointCloud <pcl::Normal>::Ptr normals(new pcl::PointCloud <pcl::Normal>);
    computeNormal(m_target, normals);

    float last_loss = 0;
    Matrix4f last_T = Matrix4f::Identity();
    // 迭代计算
    for (size_t iter = 0; iter < m_max_iters; iter++)
    {
        //查找匹配点
        std::vector<PointT> matchInput, matchTarget;
        searchKNN(output, normals,1, matchInput, matchTarget);

        if (matchInput.size() < 10) break;

        //计算RT
        Matrix4f T = Matrix4f::Identity();
        float loss = computeRT(matchInput, matchTarget, normals, T);
        if (abs(last_loss - loss) < 1e-6) break;

        last_loss = loss;
        //对点云进行变换
        pcl::transformPointCloud(output, output, T);
        //保留变换结果
        m_T = T * m_T;
        cout << "iter = " << iter << " match points size = " << matchInput.size() << " loss = " << loss  << endl;

    }
}

float  PLICP::computeRT(std::vector<PointT>& input, std::vector<PointT>& target, pcl::PointCloud <pcl::Normal>::Ptr& normals, Eigen::Matrix4f& T)
{
    int N = input.size();

    Eigen::MatrixXf A(N, 6);
    Eigen::VectorXf b(N);

    for (size_t i = 0; i < N; i++)
    {
        float x_pc1 = target[i].x;
        float y_pc1 = target[i].y;
        float z_pc1 = target[i].z;

        float x_pc2 = input[i].x;
        float y_pc2 = input[i].y;
        float z_pc2 = input[i].z;

        float nx_pc1 = normals->points[i].normal_x;
        float ny_pc1 = normals->points[i].normal_y;
        float nz_pc1 = normals->points[i].normal_z;

        A(i, 0) = -z_pc2 * ny_pc1 + y_pc2 * nz_pc1;
        A(i, 1) = z_pc2 * nx_pc1 - x_pc2 * nz_pc1;
        A(i, 2) = -y_pc2 * nx_pc1 + x_pc2 * ny_pc1;
        A(i, 3) = nx_pc1;
        A(i, 4) = ny_pc1;
        A(i, 5) = nz_pc1;

        b(i) = nx_pc1 * (x_pc1 - x_pc2) + ny_pc1 * (y_pc1 - y_pc2) + nz_pc1 * (z_pc1 - z_pc2);
    }

    Eigen::Matrix<float, 6, 1> x{ A.bdcSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(b) };

    double alpha1{ x(0) };
    double alpha2{ x(1) };
    double alpha3{ x(2) };
    double tx{ x(3) };
    double ty{ x(4) };
    double tz{ x(5) };

    T(0, 0) = 1;
    T(0, 1) = -alpha3;
    T(0, 2) = alpha2;
    T(0, 3) = tx;

    T(1, 0) = alpha3;
    T(1, 1) = 1;
    T(1, 2) = -alpha1;
    T(1, 3) = ty;

    T(2, 0) = -alpha2;
    T(2, 1) = alpha1;
    T(2, 2) = 1;
    T(2, 3) = tz;

    T(3, 0) = 0;
    T(3, 1) = 0;
    T(3, 2) = 0;
    T(3, 3) = 1;

    VectorXf residuals = A * x - b;
    return residuals.norm()/N;
}

int  main_plicp() {

    pcl::PointCloud<PLICP::PointT>::Ptr input_cloud(new pcl::PointCloud<PLICP::PointT>);
    if (pcl::io::loadPCDFile<pcl::PointXYZ>("E:\\3D\\input.pcd", *input_cloud) < 0)
    {
        PCL_ERROR("\a源点云文件不存在！\n");
        return -1;
    }
    pcl::PointCloud<PLICP::PointT>::Ptr target_cloud(new pcl::PointCloud<PLICP::PointT>);
    if (pcl::io::loadPCDFile<pcl::PointXYZ>("E:\\3D\\target.pcd", *target_cloud) < 0)
    {
        PCL_ERROR("\a目标点云文件不存在！\n");
        return -1;
    }

    system("pause");
}