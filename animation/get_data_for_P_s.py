from functions import *
import pickle


with open('true_underlying_w_nodes.pkl', 'rb') as file:
    loaded_variables = pickle.load(file)

ONE_HOTTED_GT = loaded_variables['one_hotted_GT']
CLIENT_GT_WEIGHT = loaded_variables['clients_GT_weights']


def main():
    history_idx_best_candid_runs = np.empty((NUM_RUNS, NUM_ROUNDS, NUM_CLIENTS), dtype=int)

    for iter_run in range(NUM_RUNS):
        clients_ls = get_clients_list(NUM_FEATURES, CLIENT_GT_WEIGHT)
        total_loss, mat_history_rewards, mat_history_idx_best_candid, hist_loss = algorithm(clients_ls,
                                                                                            NUM_FEATURES,
                                                                                            NUM_ROUNDS,
                                                                                            LEARNING_RATE)
        history_idx_best_candid_runs[iter_run, :, :] = mat_history_idx_best_candid
    return history_idx_best_candid_runs


history_idx_best_candid_runs = main()

with open('history_idx_best_candid_runs.pkl', 'wb') as file:
    saved_variables = {
        'history_idx_best_candid_runs': history_idx_best_candid_runs
    }
    pickle.dump(saved_variables, file)
