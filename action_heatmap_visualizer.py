import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os
import pickle
import torch
from scipy.stats import pearsonr

class ActionHeatmapVisualizer:
    def __init__(self, save_dir):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        self.acts = []
        self.outs = []
        self.ts = []
        self.names = []
        self.ok = False

    def add(self, t, acts, outs=None):
        self.ts.append(t)
        self.acts.append(acts)
        if outs is not None:
            self.outs.append(outs)
        if not self.names:
            self.names = [f'A{i+1}' for i in range(len(acts))]

    def get_changes(self):
        x = np.array(self.acts)
        d = np.zeros_like(x)
        d[1:] = np.abs(x[1:] - x[:-1])
        return d

    def get_corr(self):
        x = np.array(self.acts)
        n = x.shape[1]
        c = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                c[i, j], _ = pearsonr(x[:, i], x[:, j])
        return c

    def plot(self, task, ep, bounds=None):
        if not self.acts:
            print(f"No data for ep {ep}")
            return None, None
        fig = plt.figure(figsize=(20, 12))
        gs = fig.add_gridspec(2, 2, height_ratios=[3, 1])
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[0, 1])
        ax3 = fig.add_subplot(gs[1, :])
        d = self.get_changes()
        sns.heatmap(d.T, ax=ax1, cmap='YlOrRd', xticklabels=50, yticklabels=self.names)
        ax1.set_title(f'Changes - Ep {ep}')
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Action')
        if self.outs:
            o = np.array(self.outs)
            for t in range(len(self.ts)):
                for i in range(len(self.names)):
                    if o[t, i]:
                        ax1.add_patch(plt.Rectangle((t, i), 1, 1, fill=False, edgecolor='blue', linewidth=1))
        c = self.get_corr()
        sns.heatmap(c, ax=ax2, cmap='coolwarm', vmin=-1, vmax=1, center=0,
                   xticklabels=self.names, yticklabels=self.names)
        ax2.set_title('Correlations')
        m = np.linalg.norm(d, axis=1)
        ax3.plot(self.ts, m, 'b-')
        ax3.set_title('Magnitude')
        ax3.set_xlabel('Time')
        ax3.set_ylabel('Size')
        if bounds:
            for b in bounds:
                ax1.axvline(x=b, color='white', linestyle='--', alpha=0.5)
                ax3.axvline(x=b, color='gray', linestyle='--', alpha=0.5)
        s = f'Stats:\n'
        s += f'Time: {len(self.ts)}\n'
        s += f'Acts: {len(self.names)}\n'
        if self.outs:
            n = np.sum(np.array(self.outs))
            s += f'Outs: {n}\n'
            s += f'Out %: {n/(len(self.ts)*len(self.names))*100:.1f}%\n'
        s += f'OK: {self.ok}'
        plt.figtext(0.02, 0.98, s, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        plt.tight_layout()
        ok_dir = os.path.join(self.save_dir, 'success')
        fail_dir = os.path.join(self.save_dir, 'failure')
        os.makedirs(ok_dir, exist_ok=True)
        os.makedirs(fail_dir, exist_ok=True)
        t = datetime.now().strftime("%Y%m%d_%H%M%S")
        p = f'heatmap_{task}_ep{ep}_{t}.png'
        p_path = os.path.join(ok_dir if self.ok else fail_dir, p)
        plt.savefig(p_path, dpi=300, bbox_inches='tight')
        plt.close()
        data = {
            'acts': self.acts,
            'outs': self.outs if self.outs else None,
            'ts': self.ts,
            'names': self.names,
            'task': task,
            'ep': ep,
            'bounds': bounds,
            'stats': {
                'time': len(self.ts),
                'acts': len(self.names),
                'outs': np.sum(np.array(self.outs)) if self.outs else 0
            },
            'ok': self.ok
        }
        d = f'data_{task}_ep{ep}_{t}.pkl'
        d_path = os.path.join(self.save_dir, d)
        with open(d_path, 'wb') as f:
            pickle.dump(data, f)
        return p_path, d_path

    def set_ok(self, ok):
        self.ok = ok

    def clear(self):
        self.acts = []
        self.outs = []
        self.ts = []
        self.names = []
        self.ok = False 