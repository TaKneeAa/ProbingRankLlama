import numpy as np
import os
import torch
import argparse
import datasets
from sklearn.linear_model import LogisticRegression, LinearRegression, Lasso,Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score
from generate_labels import compute_metrics
from sequences import load_ms_marco_data, IDContext
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt


#set manually
n_queries = 98
n_docs = 50
n_layers = 32
n_dim = 4096

#load all queries from ms_marco
query_set = load_ms_marco_data(n_queries,n_docs)
feature_set = compute_metrics(query_set)

context_neurons = []

for layer in range(n_layers-1,n_layers):
    for feature in next(iter(feature_set.values()))[0]:
        activations = np.zeros((n_queries*n_docs, n_dim), dtype=float)
        labels = np.zeros(n_queries*n_docs,dtype=float)

        #load activations
        for i in range(n_queries):
            for j in range(n_docs):
                name = f'91activations/q{i}/d{j}layer_{layer}_activations.pt'
                if os.path.exists(name):
                    activation_tensor = torch.load(name)
                    activation_tensor = activation_tensor.cpu()
                    activation_np = activation_tensor.numpy()
                    activations[i*n_docs+j] = activation_np
                else:
                    break

        #load desired feature labels
        for i, query in enumerate(query_set):
            doc_list = query_set[query]
            metrics = feature_set[query]
            for j in range(n_docs):
                labels[i*n_docs+j] = metrics[j][feature]

        #regression
        X_train, X_test, y_train, y_test = train_test_split(activations, labels, test_size=0.25, random_state=42)
        model = Ridge(alpha=10.0)

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        if((r2>-10.0)):
            neuron = {}
            neuron["weights"] = model.coef_
            neuron["layer"] = layer 
            neuron["score"] = r2 
            neuron["mse"] = mse
            neuron["feature"]=feature
            context_neurons.append(neuron)
            print(layer,feature, r2)

  





    