import numpy as np
import time

def euclidean_distance(point1, point2):
    return np.sqrt(np.sum((point1 - point2) ** 2))

def find_neighbors(data, point_index, epsilon):
    neighbors = []
    for i in range(len(data)):
        if euclidean_distance(data[point_index], data[i]) <= epsilon:
            neighbors.append(i)
    return neighbors

def dbscan(data, epsilon, min_points):
    labels = [0] * len(data)
    cluster_id = 0
    for i in range(len(data)):
        if labels[i] != 0:
            continue
        neighbors = find_neighbors(data, i, epsilon)
        if len(neighbors) < min_points:
            labels[i] = -1  # Noise point
        else:
            cluster_id += 1
            expand_cluster(data, labels, i, neighbors, cluster_id, epsilon, min_points)
    return labels

def expand_cluster(data, labels, point_index, neighbors, cluster_id, epsilon, min_points):
    labels[point_index] = cluster_id
    i = 0
    while i < len(neighbors):
        point = neighbors[i]
        if labels[point] == -1:  # Change noise point to border point
            labels[point] = cluster_id
        elif labels[point] == 0:  # Unvisited point
            labels[point] = cluster_id
            new_neighbors = find_neighbors(data, point, epsilon)
            if len(new_neighbors) >= min_points:
                neighbors += new_neighbors
        i += 1

start_time = time.time()

run_times = 1
epsilon = 0.1
min_points = 4
for i in range(run_times):       
    # 生成10000个介于0和10之间的随机点
    data = np.random.uniform(0, 10, (1000, 2))
    # 运行DBSCAN算法并统计时间
    labels = dbscan(data, epsilon, min_points)
    
end_time = time.time()
total_time = end_time - start_time
print("聚类时间：", total_time, "秒")


# 10000 934s
# 20000 3509.770181417465s
