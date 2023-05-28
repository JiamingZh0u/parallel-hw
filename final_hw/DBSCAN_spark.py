from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, DoubleType
from pyspark.sql.functions import udf
from math import sqrt
import numpy as np
import time

# 初始化Spark
sc = SparkContext()
spark = SparkSession.builder.getOrCreate()

# DBSCAN算法实现
def dbscan(data, eps, min_pts):
    def distance(p1, p2):
        return sqrt(sum([(a - b) ** 2 for a, b in zip(p1, p2)]))
    
    def region_query(point):
        neighbors = []
        for p in data:
            if distance(point, p) <= eps:
                neighbors.append(p)
        return neighbors
    
    def expand_cluster(point, neighbors, cluster_id):
        cluster[point] = cluster_id
        i = 0
        while i < len(neighbors):
            neighbor = neighbors[i]
            if neighbor not in visited:
                visited.add(neighbor)
                new_neighbors = region_query(neighbor)
                if len(new_neighbors) >= min_pts:
                    neighbors.extend(new_neighbors)
            if neighbor not in cluster:
                cluster[neighbor] = cluster_id
            i += 1
    
    cluster_id = 0
    visited = set()
    cluster = {}
    for point in data:
        if point in visited:
            continue
        visited.add(point)
        neighbors = region_query(point)
        if len(neighbors) < min_pts:
            cluster[point] = -1  # 噪声点
        else:
            cluster_id += 1
            expand_cluster(point, neighbors, cluster_id)
    return cluster



start_time = time.time()

run_times = 1
eps = 0.1
min_pts = 4
for i in range(run_times):       
    # 创建示例数据
    data = np.random.uniform(0, 10, (5000, 2)).tolist()

    # 将数据转换为Spark DataFrame
    schema = StructType([StructField("x", DoubleType(), True), StructField("y", DoubleType(), True)])
    df = spark.createDataFrame(data, schema)

    # 提取数据列
    points = df.select("x", "y").rdd.map(tuple)

    # 使用DBSCAN算法进行聚类
    clusters = dbscan(points.collect(), eps, min_pts)

    # # 输出结果
    # for point, cluster_id in clusters.items():
    #     print("Point:", point, "Cluster ID:", cluster_id)
    
end_time = time.time()
total_time = end_time - start_time
print("聚类时间：", total_time, "秒")

# 10000 177s
# 20000 648s
