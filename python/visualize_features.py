import matplotlib.pyplot as plt
import numpy as np
import sys

with open(sys.argv[1], 'r') as f:
    lines = f.readlines()

features_mat = None
for line in lines:
    feature_i = np.array(map(float, line[1:-2].split(',')))
    if features_mat is not None:
        features_mat = np.vstack((features_mat, feature_i))
    else:
        features_mat = feature_i

print features_mat

for i in range(73):
    plt.plot(features_mat[:, i*128:(i+1)*128].transpose())
    plt.show()
