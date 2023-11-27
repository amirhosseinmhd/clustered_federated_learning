import torch
import numpy as np
import pickle

from ps_functions import Client
from ps_config import *


def create_true_underlying_w(num_features):
    one_hotted_GT = torch.from_numpy(np.eye(NUM_CLUSTERS)[np.random.choice(NUM_CLUSTERS,
                                                                           NUM_CLIENTS)].T)  # creates a matrix and indicates which client belongs to which cluster it has k rows and n coloumns
    core_GT_weights = torch.from_numpy(np.random.uniform(low=-10, high=10, size=(num_features, NUM_CLUSTERS)))  # (d, k)
    clients_GT_weights = core_GT_weights @ one_hotted_GT  # (d, n) = (d, k) @ (k, n)

    return one_hotted_GT, clients_GT_weights


one_hotted_GT, clients_GT_weights = create_true_underlying_w(NUM_FEATURES)
with open('true_underlying_w_nodes.pkl', 'wb') as file:
    saved_variables = {
        'one_hotted_GT': one_hotted_GT,
        'clients_GT_weights': clients_GT_weights
    }
    pickle.dump(saved_variables, file)
