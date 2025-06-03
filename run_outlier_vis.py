import torch
import numpy as np
from outlier_visualizer import OutlierVisualizer, get_subtask_boundaries

# Create visualizer
save_dir = "visualizations"
visualizer = OutlierVisualizer(save_dir)
visualizer.set_threshold(1.5)  # You can change this threshold

# Simulated rollout (10 timesteps, 5 actions each timestep)
T = 10  # timesteps
A = 5   # actions per timestep
rng = np.random.default_rng(0)

for t in range(T):
    dists = torch.tensor(rng.normal(loc=1.0, scale=0.3, size=A), dtype=torch.float32)
    
    # Compute IQR
    q1 = torch.quantile(dists, 0.25).item()
    q3 = torch.quantile(dists, 0.75).item()
    iqr = q3 - q1
    upper = q3 + 1.5 * iqr
    lower = q1 - 1.5 * iqr
    
    # Find outliers
    outliers = (dists > upper) | (dists < lower)
    outlier_indices = torch.where(outliers)[0]

    visualizer.collect_timestep_data(dists, outlier_indices, q1, q3, iqr)

# Optional: get subtask boundaries
subtasks = get_subtask_boundaries("sim_transfer_cube_human")

# Plot and save
plot_path, data_path = visualizer.plot_mahalanobis_timeseries(
    task_name="sim_transfer_cube_human",
    episode_idx=0,
    subtask_boundaries=subtasks
)

print(f"Plot saved to: {plot_path}")
print(f"Data saved to: {data_path}")
