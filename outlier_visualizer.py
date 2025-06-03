import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
from datetime import datetime
import torch

class OutlierVisualizer:
    def __init__(self, save_dir):
        self.save_dir = save_dir
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        self.mbis_distances = []
        self.removed_indices = []
        self.q1_values = []
        self.q3_values = []
        self.iqr_values = []
        self.threshold = None

    def collect_timestep_data(self, mbis_dists, removed_idx, q1, q3, iqr):
        if not isinstance(mbis_dists, torch.Tensor):
            raise TypeError("mbis_dists must be a torch.Tensor")
        if removed_idx is not None and not isinstance(removed_idx, torch.Tensor):
            raise TypeError("removed_idx must be a torch.Tensor or None")
        if not isinstance(q1, (float, int)):
            raise TypeError("q1 must be a float or int")
        if not isinstance(q3, (float, int)):
            raise TypeError("q3 must be a float or int")
        if not isinstance(iqr, (float, int)):
            raise TypeError("iqr must be a float or int")

        self.mbis_distances.append(mbis_dists.cpu().numpy())
        self.removed_indices.append(removed_idx.cpu().numpy() if removed_idx is not None else np.array([]))
        self.q1_values.append(q1)
        self.q3_values.append(q3)
        self.iqr_values.append(iqr)

    def set_threshold(self, threshold):
        self.threshold = threshold

    def clear_data(self):
        self.mbis_distances = []
        self.removed_indices = []
        self.q1_values = []
        self.q3_values = []
        self.iqr_values = []

    def plot_mahalanobis_timeseries(self, task_name, episode_idx, subtask_boundaries=None):
        plt.figure(figsize=(15, 8))
        mbis_dists = np.array(self.mbis_distances)
        q1_array = np.array(self.q1_values)
        q3_array = np.array(self.q3_values)
        iqr_array = np.array(self.iqr_values)

        for t in range(len(mbis_dists)):
            distances = mbis_dists[t]
            removed = self.removed_indices[t]
            inlier_mask = ~np.isin(np.arange(len(distances)), removed)
            plt.scatter(np.full_like(distances[inlier_mask], t), distances[inlier_mask],
                        c='gray', alpha=0.3, s=10, label='Inlier' if t == 0 else "")
            if len(removed) > 0:
                plt.scatter(np.full_like(distances[removed], t), distances[removed],
                            c='red', alpha=0.5, s=15, label='Outlier' if t == 0 else "")

        lower_bounds = q1_array - self.threshold * iqr_array
        upper_bounds = q3_array + self.threshold * iqr_array

        plt.plot(lower_bounds, 'k--', alpha=0.5, label='Lower Bound')
        plt.plot(upper_bounds, 'k--', alpha=0.5, label='Upper Bound')

        if subtask_boundaries:
            for t, label in subtask_boundaries:
                plt.axvline(x=t, color='blue', linestyle=':', alpha=0.5)
                plt.text(t, plt.ylim()[1], label, rotation=90,
                         verticalalignment='bottom', color='blue')

        plt.xlabel('Timestep')
        plt.ylabel('Mahalanobis Distance')
        plt.title(f'Mahalanobis Distance Time Series\n{task_name} - Episode {episode_idx}')
        plt.legend()

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_path = os.path.join(self.save_dir, f'mahalanobis_{task_name}_ep{episode_idx}_{timestamp}.png')
        plt.savefig(plot_path)
        plt.close()

        data = {
            'mbis_distances': mbis_dists,
            'removed_indices': self.removed_indices,
            'q1_values': q1_array,
            'q3_values': q3_array,
            'iqr_values': iqr_array,
            'threshold': self.threshold,
            'subtask_boundaries': subtask_boundaries
        }
        data_path = os.path.join(self.save_dir, f'mahalanobis_data_{task_name}_ep{episode_idx}_{timestamp}.pkl')
        with open(data_path, 'wb') as f:
            pickle.dump(data, f)

        return plot_path, data_path

def get_subtask_boundaries(task_name):
    if 'transfer_cube' in task_name:
        return [
            (50, "Pick"),
            (100, "Transfer"),
            (150, "Place")
        ]
    elif 'insertion' in task_name:
        return [
            (50, "Approach"),
            (100, "Align"),
            (150, "Insert")
        ]
    return []
