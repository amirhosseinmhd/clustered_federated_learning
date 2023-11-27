import numpy as np
import torch

from ps_config import *


class Client:
    def __init__(self, num_samples, num_features, weights, true_w, STD):
        self.weights = weights

        self.X, self.y = generate_data(num_samples, num_features,
                                       true_w, STD)
        self.GT_w = true_w

    def get_local_loss(self, weights):
        prediction = self.X @ weights
        loss_ = ((prediction - self.y) ** 2).mean()
        return loss_

    def train_locally(self, L, lr):
        weights = self.weights.clone()
        for l in range(0, L):
            weights.requires_grad_()
            weights.grad = None
            prediction = self.X @ weights
            loss_ = ((prediction - self.y) ** 2).mean()
            loss_.backward()
            grad = weights.grad.clone()
            with torch.no_grad():
                weights = weights - lr * grad
        self.weights = weights.detach()
        return weights.detach()

    def eval(self, weights):
        with torch.no_grad():
            return ((self.GT_w - weights).norm(p=2)).item()


def generate_data(num_samples, num_features, true_w, STD):
    """Generate synthetic data for a user."""
    X = torch.randn(num_samples, num_features, dtype=torch.float64)
    y = X @ true_w + STD * torch.randn(num_samples, 1)
    return X, y


def get_clients_list(num_features, clients_GT_weights):
    w_init_mat = torch.randn(num_features, NUM_CLIENTS, requires_grad=False, dtype=torch.float64)
    clients_list = []
    for client_iter in range(NUM_CLIENTS):
        GT_w = clients_GT_weights[:, client_iter].reshape(-1, 1)
        w = w_init_mat[:, client_iter].reshape(-1, 1).clone()
        clients_list.append(Client(NUM_SAMPLES, num_features, w, GT_w, STD))
    return clients_list


def find_w_candidate(current_weight, X, y, lr):
    current_weight.requires_grad_()
    current_weight.grad = None
    prediction = X @ current_weight
    loss_ = ((prediction - y) ** 2).mean()
    loss_.backward()
    with torch.no_grad():
        w_candidate = current_weight - lr * current_weight.grad
    return w_candidate


def get_w_candidates(clients_list, curr_client_id, num_features, client_ids, lr):
    curr_weight = clients_list[curr_client_id].weights.clone()
    candidates = torch.zeros((num_features, SUBSET_SIZE), dtype=torch.float64)
    for iter_client in range(SUBSET_SIZE):
        id_ = client_ids[iter_client]
        client = clients_list[id_]
        w_iter = find_w_candidate(curr_weight.clone(), client.X, client.y, lr)
        candidates[:, iter_client] = w_iter.clone().squeeze()
    return candidates


def get_w_with_best_reward(num_features, candidates, client_ids, client):
    current_loss = client.get_local_loss(client.weights)
    rewards = torch.empty((SUBSET_SIZE), dtype=torch.float64)
    for iter_client in range(SUBSET_SIZE):
        rewards[iter_client] = current_loss - client.get_local_loss(candidates[:, iter_client].reshape(num_features, 1))
    idx_max_reward = torch.argmax(rewards)
    return candidates[:, idx_max_reward], client_ids[idx_max_reward].item(), rewards[idx_max_reward].item()


def get_random_indices_without_i():
    indices = np.arange(0, NUM_CLIENTS)
    total_client_ids = np.empty((SUBSET_SIZE, NUM_CLIENTS), dtype=int)

    for iter_client in range(NUM_CLIENTS):
        available_indices = np.delete(indices, iter_client)
        random_indices = np.random.choice(available_indices, size=SUBSET_SIZE, replace=False)
        total_client_ids[:, iter_client] = random_indices
    return total_client_ids


def algorithm(clients_list, num_features, num_rounds, lr):
    total_loss = []
    mat_history_rewards = np.empty((num_rounds, NUM_CLIENTS))
    mat_history_idx_best_candid = np.empty((num_rounds, NUM_CLIENTS), dtype=int)
    hist_loss = []

    for round_ in range(num_rounds):
        total_subsets = get_random_indices_without_i()
        w_mat = torch.empty((num_features, NUM_CLIENTS), dtype=torch.float64)

        for iter_client in range(NUM_CLIENTS):
            curr_client_candidates = get_w_candidates(clients_list, iter_client, num_features,
                                                      total_subsets[:, iter_client], lr)

            w_mat[:, iter_client], mat_history_idx_best_candid[round_, iter_client], mat_history_rewards[round_,
            iter_client] = get_w_with_best_reward(num_features,
                                                  curr_client_candidates, total_subsets[:, iter_client],
                                                  clients_list[iter_client])

        loss = []

        for iter_client in range(NUM_CLIENTS):
            loss.append(clients_list[iter_client].eval(clients_list[iter_client].weights.detach()))
            clients_list[iter_client].weights = w_mat[:, iter_client].clone().reshape(-1, 1)

        total_loss.append(sum(loss) / len(loss))
        hist_loss.append(loss)
    return total_loss, mat_history_rewards, mat_history_idx_best_candid, hist_loss
