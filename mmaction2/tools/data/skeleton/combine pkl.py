import os
import pickle
import osp
from collections import Counter

path = '/home/ciis-compnew/Desktop/ciis_dataset/Thermal/SEMINAR - SKIRPSI/Dataset/Dataset - 90 10/Thr_05/PKL Train/'

custom_dataset_val = []
for d in os.listdir(path):
    if d.endswith('.pkl'):
        with open(os.path.join(path, d), 'rb') as f:
            content = pickle.load(f)
        custom_dataset_val.append(content)
with open('/home/ciis-compnew/Desktop/ciis_dataset/Thermal/SEMINAR - SKIRPSI/Combined PKL/90:10_05/custom_dataset_train_thermal.pkl', 'wb') as out:
    pickle.dump(custom_dataset_val, out, protocol=pickle.HIGHEST_PROTOCOL)