import numpy as np
import os
import random
import scipy.cluster.hierarchy as sch
import matplotlib.pyplot as pl
from sklearn.cluster import KMeans

name = []
data = []
labels = []
spr = [] #seperators

# grab all the data from .txt files in directory
for file in os.listdir("data"):
    if file.endswith(".txt") and file!="dataDescriptions.txt":
        temp_name = [line.split('^')[0] for line in open("data/"+file).read().split('\n')][:-2]
        if len(temp_name) != 0:
            name += temp_name
        temp_data = [line.split('^')[1:-2] for line in open("data/"+file).read().split('\n')][:-2]
        if len(temp_data) != 0:
            data += temp_data
            labels += [file[4:-4]]*len(temp_data)
            spr.append(len(temp_data))

data = np.array(data, dtype='float')

# normalize features
data_min = data.min(axis=0)
data_max = data.max(axis=0)

for i in range(len(data)):
    for j in range(len(data[0])):
        if data[i][j] != 0:
            data[i][j] = ((data[i][j]-data_min[j])/(data_max[j]-data_min[j]))


# set up and run clustering algorithm with 4 clusters
km = KMeans(n_clusters = 4)
km.fit(data)

def randIndex(truth, predicted):
    """
    The function is to measure similarity between two label assignments
    truth: ground truth labels for the dataset (1 x 1496)
    predicted: predicted labels (1 x 1496)
    """
    if len(truth) != len(predicted):
        print("different sizes of the label assignments")
        return -1
    elif (len(truth) == 1):
        return 1
    sizeLabel = len(truth)
    agree_same = 0
    disagree_same = 0
    count = 0
    for i in range(sizeLabel-1):
        for j in range(i+1,sizeLabel):
            if ((truth[i] == truth[j]) and (predicted[i] == predicted[j])):
                agree_same += 1
            elif ((truth[i] != truth[j]) and (predicted[i] != predicted[j])):
                disagree_same +=1
            count += 1
    return (agree_same+disagree_same)/float(count)

# generate baseline score by comparing against randomized labels
print('Baseline randIndex score: %.8f'%(randIndex(labels, np.random.permutation(labels))))

# evaluate performance of k-means clustering
print('k-means randIndex score: %.8f'%(randIndex(labels, km.predict(data))))


# since initial cluster centers are chosen at random
# run k-means on data 20 times and compare performance
funcs = [KMeans(init='random', n_init=1, n_clusters=4)]*20
funcs_perform = []
for func in funcs:
    obj = func.fit(data)
    funcs_perform.append((obj.inertia_, randIndex(labels, obj.predict(data))))
funcs_perform = [x for x in set(funcs_perform)]
min_func_index = np.argmin([x[0] for x in funcs_perform])

print('%-11s  %11s' % ('Objective', 'RandIndex'))
for i in range(len(funcs_perform)):
    if i != min_func_index:
        print('%-11f  %11f' % (funcs_perform[i][0], funcs_perform[i][1]))
    else:
        print('%-11f  %11f <-- Minimum Objective' % (funcs_perform[i][0], funcs_perform[i][1]))

# run k-means with different number of clusters and display
# some descriptive facts about the results
from collections import Counter
print("(cluster number, size)")
cluster_sizes = [5, 10, 25, 50, 75]
for s in cluster_sizes:
    print("\nNumber of clusters: "+str(s))
    km_k = KMeans(n_clusters=s)
    results = km_k.fit_predict(data[0:spr[0]])
    biggest = Counter(results).most_common()[0]
    print("Size of largest cluster: "+str(biggest[1]))
    if biggest[1] < 10:
        for i in range(len(data)):
            if results[i]==biggest[0]:
                print(name[i])
    else:
        biggest_cluster = [i for i, x in enumerate(results) if x==biggest[0]]
        random_names = [x for x in random.sample(range(len(biggest_cluster)), 10)]
        print(np.array(name)[random_names])


# visualizations of dendogram and flcuster results
fig = pl.figure(figsize=(20, 10), dpi=220) 
hClsMat = sch.linkage(data, method='complete') # Complete clustering
sch.dendrogram(hClsMat, labels=labels, leaf_rotation = 90)
fig.show()
resultingClusters = sch.fcluster(hClsMat2, t=3.8, criterion ='distance')
print(set(resultingClusters))

# evaluate performance of fcluster
print('fcluster randIndex score: %.8f'%(randIndex(resultingClusters, labels)))

# evaluate performance of fcluster with different t settings
fcluster_results = []
ti = float(0)
for i in range(20):
    ti += 0.2
    resultingClusters2 = sch.fcluster(hClsMat2, t=ti, criterion ='distance')
    r = (ti, randIndex(resultingClusters2, labels))
    fcluster_results.append(r)
    print('t=%.1f  randIndex=%9f' % r)

# find and print best fcluster result and respective parameters
best_fcluster_param =  max(fcluster_results, key = lambda t: t[1])
best_fcluster_results = sch.fcluster(hClsMat2, t=best_fcluster_param[0], criterion ='distance')
best_fcluster_num_cluster = len(set(best_fcluster_results))
print('Best results for threshold=%.1f; RI=%9f, Number of clusters=%d' % (best_fcluster_param[0], randIndex(best_fcluster_results, labels), best_fcluster_num_cluster))


