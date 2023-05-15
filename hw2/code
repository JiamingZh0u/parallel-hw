#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <fstream>
#include <immintrin.h>
#include <pthread.h>

using namespace std;

const double eps = 0.1;  // 邻域半径
const int minPts = 4;   // 最小密度阈值

struct Point {
    double x, y;
    bool visited;
    int cluster;
    Point(double _x, double _y) : x(_x), y(_y), visited(false), cluster(0) {}
};

double dist(const Point& p1, const Point& p2) {
    __m128d xmm1, xmm2, xmm;
    double temp1[2] __attribute__((aligned(16))) = {p1.y, p1.x};
    double temp2[2] __attribute__((aligned(16))) = {p2.y, p2.x};
    xmm1 = _mm_load_pd(temp1);
    xmm2 = _mm_load_pd(temp2);
    xmm = _mm_sub_pd(xmm1, xmm2);
    xmm = _mm_mul_pd(xmm, xmm);
    xmm = _mm_hadd_pd(xmm, xmm);
    double result;
    _mm_store_sd(&result, xmm);
    return sqrt(result);
}

// 密度可达
bool isDensityReachable(const Point& p1, const Point& p2) {
    return dist(p1, p2) <= eps;
}

// 获取点 p 的邻域
vector<Point*> getNeighborhood(const vector<Point>& points, const Point& p) {
    vector<Point*> neighborhood;
    for (auto& q : points) {
        if (isDensityReachable(p, q)) {
            neighborhood.push_back(const_cast<Point*>(&q));
        }
    }
    return neighborhood;
}

// 判断是否为核心点
bool isCorePoint(const vector<Point>& points, const Point& p) {
    return getNeighborhood(points, p).size() >= minPts;
}

// 将点 p 添加到聚类 c 中
void addToCluster(vector<Point>& points, Point& p, int c) {
    p.cluster = c;
    for (auto& q : getNeighborhood(points, p)) {
        if (!q->visited) {
            q->visited = true;
            if (isCorePoint(points, *q)) {
                addToCluster(points, *q, c);
            } else {
                q->cluster = c;
            }
        }
    }
}

// DBSCAN 算法，处理子集
void* dbscan_subset(void* args) {
    auto* args_ = static_cast<pair<vector<Point>, int>*>(args);
    auto points = args_->first;
    auto start_idx = args_->second;
    auto end_idx = start_idx + points.size() / 4;

    for (int i = start_idx; i < end_idx; i++) {
        auto& p = points[i];
        if (!p.visited) {
            p.visited = true;
            if (isCorePoint(points, p)) {
                int c = i;
                addToCluster(points, p, c);
            }
            else {
                p.cluster = -1;
            }
        }
    }
    pthread_exit(NULL);
}

// Pthread 传递参数
struct Args {
    vector<Point>* points;
    int startIndex;
    int endIndex;
    int* clusterCount;
    pthread_mutex_t* mutex;
};

// Pthread 线程函数
void* threadFunc(void* args) {
    Args* arg = static_cast<Args*>(args);
    vector<Point>& points = *(arg->points);
    int startIndex = arg->startIndex;
    int endIndex = arg->endIndex;
    int clusterCount = *(arg->clusterCount);
    pthread_mutex_t* mutex = arg->mutex;
    for (int i = startIndex; i < endIndex; i++) {
        Point& p = points[i];
        if (!p.visited) {
            p.visited = true;
            if (isCorePoint(points, p)) {
                int c;
                pthread_mutex_lock(mutex);
                c = ++clusterCount;
                pthread_mutex_unlock(mutex);
                addToCluster(points, p, c);
            } else {
                p.cluster = -1;
            }
        }
    }
    pthread_exit(NULL);
}

// 并行 DBSCAN 算法
void parallelDbscan(vector<Point>& points) {
    const int numThreads = 4;
    int clusterCount = 0;
    pthread_t threads[numThreads];
    pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;
    vector<Args> args(numThreads);
    int numPointsPerThread = points.size() / numThreads;
    for (int i = 0; i < numThreads; i++) {
        args[i].points = &points;
        args[i].startIndex = i * numPointsPerThread;
        args[i].endIndex = (i == numThreads - 1) ? points.size() : (i + 1) * numPointsPerThread;
        args[i].clusterCount = &clusterCount;
        args[i].mutex = &mutex;
        pthread_create(&threads[i], NULL, threadFunc, &args[i]);
    }
    for (int i = 0; i < numThreads; i++) {
        pthread_join(threads[i], NULL);
    }
}

int main() {
    for (int i = 0; i < 20; i++) {
        vector<Point> points;
        default_random_engine e;
        uniform_real_distribution<double> u(0, 10);
        ofstream fout("result.txt"); // 打开文件用于保存运行结果

        // 生成随机点
        for (int i = 0; i < 20000; i++) {
            points.emplace_back(u(e), u(e));
        }

        //dbscan(points);
        parallelDbscan(points);

        // 输出每个点所属的聚类
        for (auto& p : points) {
            fout << "(" << p.x << ", " << p.y << ") -> cluster " << p.cluster << endl;
        }

        fout.close();  // 关闭文件
        cout << "method_2 run successfully" << endl;
    }
    return 0;
}

