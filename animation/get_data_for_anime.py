from config import *
from functions import *
import pickle

with open('true_underlying_w_nodes.pkl', 'rb') as file:
    loaded_variables = pickle.load(file)

ONE_HOTTED_GT = loaded_variables['one_hotted_GT']
CLIENT_GT_WEIGHT = loaded_variables['clients_GT_weights']

clients_ls = get_clients_list(NUM_FEATURES, CLIENT_GT_WEIGHT)
total_loss, mat_history_rewards, mat_history_idx_best_candid, hist_loss = algorithm(clients_ls,
                                                                                    NUM_FEATURES,
                                                                                    NUM_ROUNDS,
                                                                                    LEARNING_RATE)

with open('saved_variables.pkl', 'wb') as file:
    saved_variables = {
        'total_loss': total_loss,
        'mat_history_rewards': mat_history_rewards,
        'mat_history_idx_best_candid': mat_history_idx_best_candid,
        'hist_loss': hist_loss,
        'one_hotted_GT': ONE_HOTTED_GT
    }
    pickle.dump(saved_variables, file)
