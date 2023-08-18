import torch
import numpy as np
import pickle
from tqdm import tqdm

dataset = "pku"
split = "split_9"

split_1 = [4,19,31,47,51]
split_2 = [12,29,32,44,59]
split_3 = [7,20,28,39,58]

split_4 = [3, 18, 26, 38, 41, 60, 87, 99, 102, 110]
split_5 = [5, 12, 14, 15, 17, 42, 67, 82, 100, 119]
split_6 = [6, 20, 27, 33, 42, 55, 71, 97, 104, 118]

split_7 = [1, 9, 20, 34, 50]
split_8 = [3, 14, 29, 31, 49]
split_9 = [2, 15, 39, 41, 43]

train_path = '../souredata/'+dataset+'_frame50/xsub/train_position.npy'
test_path = '../souredata/'+dataset+'_frame50/xsub/val_position.npy'
train_label_path = '../souredata/'+dataset+'_frame50/xsub/train_label.pkl'
test_label_path = '../souredata/'+dataset+'_frame50/xsub/val_label.pkl'

seen_train_data_path = "./data/zeroshot/"+dataset+"/"+split+"/seen_train_data.npy"
seen_train_label_path = "./data/zeroshot/"+dataset+"/"+split+"/seen_train_label.npy"
seen_test_data_path = "./data/zeroshot/"+dataset+"/"+split+"/seen_test_data.npy"
seen_test_label_path = "./data/zeroshot/"+dataset+"/"+split+"/seen_test_label.npy"
unseen_data_path = "./data/zeroshot/"+dataset+"/"+split+"/unseen_data.npy"
unseen_label_path = "./data/zeroshot/"+dataset+"/"+split+"/unseen_label.npy"

with open(train_label_path, 'rb') as f:
    _, train_label = pickle.load(f)

with open(test_label_path, 'rb') as f:
    _, test_label = pickle.load(f)

train_data = np.load(train_path)
test_data = np.load(test_path)

print("train size:",train_data.shape)
print("test size:",test_data.shape)

seen_train_data = []
seen_train_label = []
seen_test_data = []
seen_test_label = []
unseen_data = []
unseen_label = []

for i in range(len(train_label)):
    if train_label[i] not in eval(split):
        seen_train_label.append(train_label[i])
        seen_train_data.append(train_data[i])

for i in range(len(test_label)):
    if test_label[i] in eval(split):
        unseen_label.append(test_label[i])
        unseen_data.append(test_data[i])
    else:
        seen_test_label.append(test_label[i])
        seen_test_data.append(test_data[i])

seen_train_data = np.array(seen_train_data)
seen_test_data = np.array(seen_test_data)
unseen_data = np.array(unseen_data)

print(seen_train_data.shape)
print(len(seen_train_label))
print(seen_test_data.shape)
print(len(seen_test_label))

np.save(seen_train_data_path, seen_train_data)
np.save(seen_train_label_path, seen_train_label)
np.save(seen_test_data_path, seen_test_data)
np.save(seen_test_label_path, seen_test_label)
np.save(unseen_data_path, unseen_data)
np.save(unseen_label_path, unseen_label)
