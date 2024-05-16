import numpy as np
import os
import torch
import datasets
import argparse
from sklearn.linear_model import LogisticRegression, LinearRegression, Lasso
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score
from generate_labels import compute_metrics
from sequences import get_input_sequences, IDContext
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

query_set = get_input_sequences()
feature_set = compute_metrics(query_set)
n_queries = len(query_set)

context_neurons = []
activations = np.zeros((n_queries*5, 4096), dtype=float)
labels = np.zeros(n_queries*5,dtype=float)

for feature in next(iter(feature_set.values()))[0]:
    flag=1
    for layer in range(32):

        #load activations
        for i in range(n_queries):
            for j in range(5):
                name = f'mean_activations/q{i}/d{j}layer_{layer}_activations.pt'
                if os.path.exists(name):
                    activation_tensor = torch.load(name)
                    activation_tensor = activation_tensor.cpu()
                    activation_np = activation_tensor.numpy()
                    activations[i*5+j] = activation_np
                else:
                    break

        #load desired feature labels
        for i, query in enumerate(query_set):
            doc_list = query_set[query]
            metrics = feature_set[query]
            for j in range(5):
                labels[i*5+j] = metrics[j][feature]


        #regression
        X_train, X_test, y_train, y_test = train_test_split(activations, labels, test_size=0.25, random_state=42)
        model = LinearRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)


        if(r2>0.4):
            flag=0

        if(feature=="BM25"):
            neuron = {}
            neuron["weights"] = model.coef_
            neuron["layer"] = layer 
            neuron["score"] = r2 
            neuron["mse"] = mse
            neuron["feature"]=feature
            context_neurons.append(neuron)
            print(layer,feature, r2)
    
    if(flag==1):
        print(feature)








    