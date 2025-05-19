import numpy as np
from Model.model_multi import DiseasesPredictor
import time
import pickle


file_path = "./data/sample_data/sample_garph"
adj_lists = pickle.load(open(file_path + ".adj.pkl", "rb"))
labels = pickle.load(open(file_path + ".label.pkl", "rb"))
train = pickle.load(open(file_path + ".train.pkl", "rb"))
test = pickle.load(open(file_path + ".test.pkl", "rb"))

"""adj_lists = pickle.load(open("adj_list.pkl", "rb"))
labels = pickle.load(open("labels.pkl", "rb"))
train = pickle.load(open("train_data.pkl", "rb"))
test = pickle.load(open("test_data.pkl", "rb"))"""


multi_class_num = 9260
feature_dim = 10000
epoch = 8000
batch_num = 200
lr = 0.3
feat_data = np.random.random((1008, feature_dim))
train_enc_dim = [1000, 1000, 1000, 1000]
t1 = time.time()
model = DiseasesPredictor(feat_data=feat_data,
                          multi_class_num=multi_class_num,
                          labels=labels,
                          adj_lists=adj_lists,
                          feature_dim=feature_dim,
                          train_enc_num=1,
                          train_enc_dim=train_enc_dim,
                          train_sample_num=[5, 5, 5, 5],
                          train=train, test=test,
                          kernel='GCN',
                          cuda=True,
                          topk=(1, 2, 3, 4, 5,10))

model.run(epoch, batch_num, lr)  # epoch, batch_num, lr
print(feature_dim, train_enc_dim)
print("running time:", time.time()-t1, "s")
