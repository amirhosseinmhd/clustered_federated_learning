import matplotlib.pyplot as plt
import torch
from matplotlib.widgets import Slider, Button
import copy
import pickle

from config import *
from functions import *

with open('saved_variables.pkl', 'rb') as file:
    loaded_variables = pickle.load(file)

loaded_total_loss = loaded_variables['total_loss']
loaded_mat_history_rewards = loaded_variables['mat_history_rewards']
loaded_mat_history_idx_best_candid = loaded_variables['mat_history_idx_best_candid']
loaded_hist_loss = loaded_variables['hist_loss']
one_hotted_GT = loaded_variables['one_hotted_GT']


def get_coordinates(one_hot_grand_truth):
    np.random.seed(42)
    list_x = [np.random.uniform(-10, -3) if one_hot_grand_truth[0, iter_client] else np.random.uniform(3, 10) for
              iter_client in range(NUM_CLIENTS)]
    x = np.array(list_x)
    y = np.random.uniform(-5, 5, 100)
    return x, y


def convert_onehot_2_scalar(one_hot):
    return torch.argmax(one_hot).item() + 1


def update(val):
    index = int(index_slider.val)
    idx_best_candid = loaded_mat_history_idx_best_candid[index, INSPECTED_NODE]
    copy_color = copy.copy(mask_colors)
    copy_color[idx_best_candid] = 'yellow'
    scatter_plot.set_facecolors(copy_color)

    curr_reward = loaded_mat_history_rewards[index, INSPECTED_NODE]
    curr_loss = loaded_hist_loss[index][INSPECTED_NODE]
    candidate_cluster = convert_onehot_2_scalar(one_hotted_GT[:, idx_best_candid])
    explanation_text = (
        f'Best candidate at {index}th round is client: {idx_best_candid}\nReward of this candidate: {round(curr_reward, 4)}\n'
        f'Current loss {round(curr_loss, 4)}\n'
        f'Cluster of candidate: {candidate_cluster}')
    annotation_text.set_text(explanation_text)

    fig.canvas.draw_idle()


def on_right_button_clicked(event):
    index_slider.set_val(index_slider.val + 1)


def on_left_button_clicked(event):
    index_slider.set_val(index_slider.val - 1)


######################################## Creating intercative plot
x, y = get_coordinates(one_hotted_GT)

mask_colors = ['blue' if one_hotted_GT[0, iter_client] else 'red' for iter_client in range(NUM_CLIENTS)]
mask_colors[INSPECTED_NODE] = 'green'

fig, ax = plt.subplots(figsize=(15, 8))  # Adjust the values as needed
plt.subplots_adjust(bottom=0.25, right=0.75)
scatter_plot = ax.scatter(x, y)
index_axis = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor="White")
index_slider = Slider(index_axis, 'Round', 0, NUM_ROUNDS - 1, valinit=0, valstep=1)

# Add right and left buttons
ax_button_right = plt.axes([0.8, 0.01, 0.1, 0.04])
ax_button_left = plt.axes([0.7, 0.01, 0.1, 0.04])
button_right = Button(ax_button_right, 'Next Round', color='lightgoldenrodyellow', hovercolor='0.975')
button_left = Button(ax_button_left, 'Prev Round', color='lightgoldenrodyellow', hovercolor='0.975')

button_right.on_clicked(on_right_button_clicked)
button_left.on_clicked(on_left_button_clicked)

annotation_text = ax.text(1.05, 0.5,
                          f'Here ID of the best candidate,\nCorresponding reward, and\nCurrent loss of inspected node (i={INSPECTED_NODE})\n'
                          f'will be shown.',
                          transform=ax.transAxes, va='center', ha='left')
inspect_node_cluster = convert_onehot_2_scalar(one_hotted_GT[:, INSPECTED_NODE])
ax.set_title(f'Examining the behaviour of PFL with Regret Minimization at Node {INSPECTED_NODE}')
ax.text(0.5, -0.1, f'Inspected client (i={INSPECTED_NODE}) is shown with green color and selected candidate for this'
                   f' node is highlighted with yellow.\n We have two clusters, first cluster in Blue and the second in Red color.',
        transform=ax.transAxes, va='center', ha='center')

scatter_plot.set_facecolors(mask_colors)

index_slider.on_changed(update)


############################################ Creating plot for the loss of the inspected node

def get_loss_inspected_node(history_losses):
    loss_inspected_node = []
    for round_ in range(NUM_ROUNDS):
        loss_inspected_node.append(history_losses[round_][INSPECTED_NODE])
    return loss_inspected_node


fig2, ax2 = plt.subplots(figsize=(8, 6))  # Adjust the values as needed
loss_inspeted_node = get_loss_inspected_node(loaded_hist_loss)
ax2.plot(np.arange(NUM_ROUNDS), loss_inspeted_node, label='Loss')

ax2.set_title("Average MSE for node:" + str(INSPECTED_NODE))
ax2.set_xlabel("Round")
ax2.set_ylabel("Avg MSE")
ax2.grid(True)

plt.show()
