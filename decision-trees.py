import numpy as np

feature_list = open('ps4_data/features.txt').read().split('\n')
for i in range(len(feature_list)):
    feature_list[i] = feature_list[i][0:-1]

train_data = []
for line in open('ps4_data/adult_train.txt').read().split('\n'):
    if line != "":
        train_data.append([x.strip() for x in line[0:-1].split(',')])

import re
features = []
for line in feature_list:
    if "continuous" not in line:
        d = {}
        l = re.split(':? ?,? *',line)
        for i in range(len(l)):
            d[l[i].strip()] = 0
        features.append(d)
    else:
        features.append(re.split(':? ?,? *',line)[0])

total_age = 0
age_count = 0
total_cap_gain = 0
cap_gain_count = 0
total_cap_loss = 0
cap_loss_count = 0
total_hrs = 0
hrs_count = 0
for element in train_data:
    # 1. Average Age
    if element[0] != '?':
        total_age += int(element[0])
        age_count += 1
    # 2. Mode Workclass
    if element[1] != '?':
        features[1][element[1]] += 1
    # 3. Mode Education
    if element[2] != '?':
        features[2][element[2]] += 1
    # 4. Mode Marital Status
    if element[3] != '?':   
        features[3][element[3]] += 1
    # 5. Mode Occupation
    if element[4] != '?':
        features[4][element[4]] += 1
    # 6. Mode Relationship
    if element[5] != '?':
        features[5][element[5]] += 1
    # 7. Mode Race
    if element[6] != '?':
        features[6][element[6]] += 1
    # 8. Mode Sex
    if element[7] != '?':
        features[7][element[7]] += 1
    # 9. Mean Capital Gain 
    if element[8] != '?':
        total_cap_gain += int(element[8])
        cap_gain_count += 1
    # 10. Mean Capital Loss
    if element[9] != '?':
        total_cap_loss += int(element[9])
        cap_loss_count += 1
    # 11. Mean Hours per Week
    if element[10] != '?':
        total_hrs += int(element[10])
        hrs_count += 1
    # 12. Mode Native Country
    if element[11] != '?':
        features[11][element[11]] += 1

default_vals = []
default_vals.append(total_age/age_count)
default_vals.append(max(features[1], key=features[1].get))
default_vals.append(max(features[2], key=features[2].get))
default_vals.append(max(features[3], key=features[3].get))
default_vals.append(max(features[4], key=features[4].get))
default_vals.append(max(features[5], key=features[5].get))
default_vals.append(max(features[6], key=features[6].get))
default_vals.append(max(features[7], key=features[7].get))
default_vals.append(total_cap_gain/cap_gain_count)
default_vals.append(total_cap_loss/cap_loss_count)
default_vals.append(total_hrs/hrs_count)
default_vals.append(max(features[11], key=features[11].get))

for element in train_data:
    for i in range(len(element)):
        if element[i] == '?':
            element[i] = default_vals[i]

# create template feature vector to handle categorical variables
new_features = []
for line in feature_list:
    if "continuous" not in line:
        l = re.split(':? ?,? *',line)
        for i in range(1, len(l)):
            new_features.append(l[0]+'::'+l[i])
    else:
        new_features.append(re.split(':? ?,? ?',line)[0])
    new_features.append('label')

def uncategorize(data, new_features):
    bin_train_data = np.zeros((len(data), len(new_features)))
    for i in range(len(data)):
        bin_train_data[i][new_features.index('age')] = data[i][0]
        bin_train_data[i][[j for j, s in enumerate(new_features) if data[i][1] in s]] = 1
        bin_train_data[i][[j for j, s in enumerate(new_features) if data[i][2] in s]] = 1
        bin_train_data[i][[j for j, s in enumerate(new_features) if data[i][3] in s]] = 1
        bin_train_data[i][[j for j, s in enumerate(new_features) if data[i][4] in s]] = 1
        bin_train_data[i][[j for j, s in enumerate(new_features) if data[i][5] in s]] = 1
        bin_train_data[i][[j for j, s in enumerate(new_features) if data[i][6] in s]] = 1
        bin_train_data[i][[j for j, s in enumerate(new_features) if data[i][7] in s]] = 1
        bin_train_data[i][new_features.index('capital-gain')] = data[i][8]
        bin_train_data[i][new_features.index('capital-loss')] = data[i][9]
        bin_train_data[i][new_features.index('hours-per-week')] = data[i][10]
        bin_train_data[i][[j for j, s in enumerate(new_features) if data[i][11] in s]] = 1
        if data[i][-1] == '<=50':
            bin_train_data[i][-1] = 1
        else:
            bin_train_data[i][-1] = 0
    return bin_train_data


train_data_bin_cont = uncategorize(train_data, new_features)

np.random.shuffle(train_data_bin_cont)
 
split_index = int(0.7*len(train_data_bin_cont))
full_train_x = train_data_bin_cont[:, :-2]
full_train_y = train_data_bin_cont[:, -1:]
train_x = train_data_bin_cont[:split_index, :-2]
train_y = train_data_bin_cont[:split_index, -1:]
validate_x = train_data_bin_cont[split_index:, :-2]
validate_y = train_data_bin_cont[split_index:, -1:]

from sklearn import tree

# == records of perormance for specified tree depth == #
train_performance_depth = np.zeros(30)
validate_performance_depth = np.zeros(30)
depth_vals = np.array([i for i in range(1, 31)])

# == records of performnace for specified minimum number of samples at leaf == #
train_performance_samples = np.zeros(50)
validate_performance_samples = np.zeros(50)
samples_vals = np.array([i for i in range(1, 51)])


def depth_performance(train_x, train_y, val_x, val_y, depths):
    for i in depths:
        cl = tree.DecisionTreeClassifier(max_depth=i)
        cl.fit(train_x, train_y)
        train_performance_depth[i-1] = cl.score(train_x, train_y)
        validate_performance_depth[i-1] = cl.score(validate_x, validate_y)

def samples_performance(train_x, train_y, val_x, val_y, depths):
    for i in depths:
        cl = tree.DecisionTreeClassifier(min_samples_leaf=i)
        cl.fit(train_x, train_y)
        train_performance_samples[i-1] = cl.score(train_x, train_y)
        validate_performance_samples[i-1] = cl.score(validate_x, validate_y)


depth_results = depth_performance(train_x, train_y, validate_x, validate_y, depth_vals)
samples_results = samples_performance(train_x, train_y, validate_x, validate_y, samples_vals)

import matplotlib.pyplot as plt
plt.rc("savefig", dpi=160)

fig1, ax1 = plt.subplots(figsize=(10, 6))

ax1.grid(True, which='both', alpha=0.25)
ax1.spines['top'].set_color('none')
ax1.spines['right'].set_color('none')
ax1.set_xlim(0, 30)
ax1.scatter(depth_vals, train_performance_depth, marker='o', color='blue', label='train performance')
ax1.scatter(depth_vals, validate_performance_depth, marker='s', color='green', label='validate performance')
plt.legend(loc='upper left')
ax1.set_xlabel('max_depth')


fig2, ax2 = plt.subplots(figsize=(10, 6))

ax2.grid(True, which='both', alpha=0.25)
ax2.spines['top'].set_color('none')
ax2.spines['right'].set_color('none')
ax2.set_xlim(0, 50)
ax2.scatter(samples_vals, train_performance_samples, marker='o', color='blue', label='train performance')
ax2.scatter(samples_vals, validate_performance_samples, marker='s', color='green', label='validate performance')
plt.legend(loc='upper left')
ax2.set_xlabel('min_samples_leaf')

cl_params_set = tree.DecisionTreeClassifier(max_depth=11, min_samples_leaf=23)
cl_params_set.fit(train_x, train_y)

from sklearn.externals.six import StringIO
with open("tree.dot", 'w') as f:
    f = tree.export_graphviz(cl_params_set, out_file=f,
                             max_depth=3,
                             feature_names=new_features,
                             filled=True, rounded=True)

test_data = []
test_labels = []
for line in open('ps4_data/adult_test.txt').read().split('\n'):
    if line != "":
        test_data.append([x.strip() for x in line[0:-2].split(',')])

for element in test_data:
    for i in range(len(element)):
        if element[i] == '?':
            element[i] = default_vals[i]

test_bin_cont = uncategorize(test_data, new_features)
test_x = test_bin_cont[:, :-2]
test_y = test_bin_cont[:, -1:]

cl = tree.DecisionTreeClassifier(max_depth=11, min_samples_leaf=23)
cl.fit(full_train_x, full_train_y)
print(cl.score(test_x, test_y))

