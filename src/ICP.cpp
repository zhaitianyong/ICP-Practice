//
// Created by atway on 2022/3/19.
//
#include "ICP.h"


ICP::ICP() {

    m_max_iters = 20;
    m_T = Matrix4f::Identity();
}

void ICP::setInputSource(PointCloudT::Ptr& source) {

    m_source = source;
}

void ICP::setInputTarget(PointCloudT::Ptr& target) {
    m_target = target;
    m_kdtree.setInputCloud(m_target);
}

void ICP::setMaximumIterations(int iterations)
{
    this->m_max_iters = iterations;
}

Eigen::Matrix4f ICP::getFinalTransformation()
{
    return m_T;
}

float ICP::getFitnessScore()
{
    return 0.0f;
}

float ICP::getLoss(std::vector<PointT>& source, std::vector<PointT>& target, Eigen::Matrix4f T)
{
    return 0.0f;
}



void ICP::searchKNN(PointCloudT& input, int k, std::vector<PointT>& matchInput, std::vector<PointT>& matchTarget)
{
    std::vector<pair<int, int>> ids;
    std::vector<float> distances;
    {
        std::vector<int> pointIdxNKNSearch(k);
        std::vector<float> pointNKNSquaredDistance(k);
        for (int i = 0; i < input.size(); i++)
        {
            if (m_kdtree.nearestKSearch(input.points[i], k, pointIdxNKNSearch, pointNKNSquaredDistance) > 0) {
                int& index = pointIdxNKNSearch[0];
                ids.push_back(make_pair(i, index));
                distances.push_back(pointNKNSquaredDistance[0]);
            }
        }
    }
    //对查找的点进行过滤
    {
        float min_planarity = 0.05;
        float sum = std::accumulate(distances.begin(), distances.end(), 0.0);
        float mean = sum / distances.size();
        double accum = 0.0;
        std::for_each(distances.begin(), distances.end(), [&](const double d) {
            accum += (d - mean) * (d - mean);
        });
        double stdev = sqrt(accum / (distances.size() - 1)); //方差

        for (auto& id : ids)
        {
            //距离判断
            if (abs(distances[id.first] - mean) > 3 * stdev)
                continue;
            matchInput.push_back(input.points[id.first]);
            matchTarget.push_back(m_target->points[id.second]);
        }

    }

}


void ICP::align(PointCloudT& output) {

    pcl::copyPointCloud(*m_source, output);
    float last_loss = 0;
    Matrix4f last_T = Matrix4f::Identity();
    for (size_t iter = 0; iter < m_max_iters; iter++)
    {
        //查找匹配点
        std::vector<PointT> matchInput, matchTarget;
        searchKNN(output, 1, matchInput, matchTarget);

        if (matchInput.size() < 10) break;

        Matrix4f T = Matrix4f::Identity();
        //开始计算svd 分解计算RT
        float loss = computeRT(matchInput, matchTarget, T);
        if (abs(last_loss - loss) < 1e-6) break;
        last_loss = loss;
        //T 变化很小，也迭代终止
        Matrix3f R = T.block(0, 0, 3, 3);
        Vector3f t;
        t << T(0, 3), T(1, 3), T(2, 3);
        Sophus::SE3f SE3_Rt(R, t);
        //转换为李代数
        MatrixXf se3 = SE3_Rt.log();
        if(se3.norm() < 0.0001) break;
        //对点云进行变换
        pcl::transformPointCloud(output, output, T);
        //保留变换结果
        m_T = T * m_T;
        last_T = T;

        //判断T的变化量很小的情况下
        cout << "iter = " << iter << " loss = " << loss  << " T " << se3.norm() << endl;

    }
}

float  ICP::computeRT(std::vector<PointT>& input, std::vector<PointT>& target, Eigen::Matrix4f& T)
{
    int N = input.size();
    MatrixXf inputMatrix(N, 3), targetMatrix(N, 3);

    for (size_t i = 0; i < N; i++)
    {
        inputMatrix(i, 0) = input[i].x;
        inputMatrix(i, 1) = input[i].y;
        inputMatrix(i, 2) = input[i].z;
        targetMatrix(i, 0) = target[i].x;
        targetMatrix(i, 1) = target[i].y;
        targetMatrix(i, 2) = target[i].z;
    }
    MatrixXf inputMean = inputMatrix.colwise().mean();
    MatrixXf targetMean = targetMatrix.colwise().mean();
    MatrixXf inputRemoveCenter = inputMatrix.rowwise() - inputMatrix.colwise().mean();
    MatrixXf targetRemoveCenter = targetMatrix.rowwise() - targetMatrix.colwise().mean();

    //构建协方差矩阵
    Matrix3f S = inputRemoveCenter.transpose() * targetRemoveCenter;
    //svd 分解
    JacobiSVD<Eigen::MatrixXf> svd(S, ComputeFullU | ComputeFullV);
    MatrixXf V = svd.matrixV(), U = svd.matrixU();

    MatrixXf R = V * U.transpose();
    if (R.determinant() <0)
    {
        R = -R;
    }

    MatrixXf t = targetMean.transpose() - R * inputMean.transpose(); //3x1

    T.block(0, 0, 3, 3) = R;
    T.block(0, 3, 3, 1) = t;

    MatrixXf cost =  R* inputMatrix.transpose() - targetMatrix.transpose();
    float sum = 0;
    for (size_t i = 0; i < N; i++)
    {
        sum += cost.col(i).dot(cost.col(i).transpose());
    }
    sum /= N;

    return sum;
}

