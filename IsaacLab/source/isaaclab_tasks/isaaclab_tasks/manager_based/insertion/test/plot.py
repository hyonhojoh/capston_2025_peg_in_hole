import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# CSV 파일 로드 경로
# Load CSV file path
path = "/home/hyunho_RCI/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/insertion/plot/data/data_1.csv"
try:
    df = pd.read_csv(path)
except FileNotFoundError:
    print(f"Error: '{path}' not found. Please make sure the CSV file exists at the specified path.")
    exit()

half_steps = len(df) // 4

net_contact_force_ee_cols = [f"obs_{i}" for i in range(41, 41 + 6)]
FT_sensor_ee_cols = [f"obs_{i}" for i in range(47, 47 + 6)]
FT_filtered_sensor_ee_cols = [f"obs_{i}" for i in range(53, 53 + 6)]

# Extract data
net_contact_force_ee_data = df[net_contact_force_ee_cols].iloc[:half_steps]
FT_sensor_ee_data = df[FT_sensor_ee_cols].iloc[:half_steps]
FT_filtered_sensor_ee_data = df[FT_filtered_sensor_ee_cols].iloc[:half_steps]

fig, axes = plt.subplots(6, 1, figsize=(12, 16), sharex=True)
fig.suptitle('Senario 1')



# 1. net_contact_force_ee - contact_force_1 (첫 3개 컴포넌트) 플롯
# Plot net_contact_force_ee - contact_force_1 (first 3 components)
axes[0].plot(net_contact_force_ee_data.index, net_contact_force_ee_data[net_contact_force_ee_cols[0]], label=f'x')
axes[0].plot(net_contact_force_ee_data.index, net_contact_force_ee_data[net_contact_force_ee_cols[1]], label=f'y')
axes[0].plot(net_contact_force_ee_data.index, net_contact_force_ee_data[net_contact_force_ee_cols[2]], label=f'z')
axes[0].set_title('contact force 1')
axes[0].set_ylabel('Force (N)')
axes[0].legend()
axes[0].grid(True)

# 2. net_contact_force_ee - contact_force_2 (다음 3개 컴포넌트) 플롯
# Plot net_contact_force_ee - contact_force_2 (next 3 components)
axes[1].plot(net_contact_force_ee_data.index, net_contact_force_ee_data[net_contact_force_ee_cols[3]], label=f'x')
axes[1].plot(net_contact_force_ee_data.index, net_contact_force_ee_data[net_contact_force_ee_cols[4]], label=f'y')
axes[1].plot(net_contact_force_ee_data.index, net_contact_force_ee_data[net_contact_force_ee_cols[5]], label=f'z')
axes[1].set_title('contact force 2')
axes[1].set_ylabel('Force (N)')
axes[1].legend()
axes[1].grid(True)

# 3. FT_filtered_sensor_ee - FT_1 (첫 3개 컴포넌트) 플롯
# Plot FT_filtered_sensor_ee - FT_1 (first 3 components)
axes[2].plot(FT_sensor_ee_data.index, FT_sensor_ee_data[FT_sensor_ee_cols[0]], label=f'F_x')
axes[2].plot(FT_sensor_ee_data.index, FT_sensor_ee_data[FT_sensor_ee_cols[1]], label=f'F_y')
axes[2].plot(FT_sensor_ee_data.index, FT_sensor_ee_data[FT_sensor_ee_cols[2]], label=f'F_z')
axes[2].set_title('Raw Force Sensor Data')
axes[2].set_ylabel('Force (N)')
axes[2].legend()
axes[2].grid(True)

# 4. FT_filtered_sensor_ee - FT_2 (다음 3개 컴포넌트) 플롯
# Plot FT_filtered_sensor_ee - FT_2 (next 3 components)
axes[3].plot(FT_sensor_ee_data.index, FT_sensor_ee_data[FT_sensor_ee_cols[3]], label=f'T_x')
axes[3].plot(FT_sensor_ee_data.index, FT_sensor_ee_data[FT_sensor_ee_cols[4]], label=f'T_y')
axes[3].plot(FT_sensor_ee_data.index, FT_sensor_ee_data[FT_sensor_ee_cols[5]], label=f'T_z')
axes[3].set_title('Raw Torque Sensor Data')
axes[3].set_xlabel('Time Step')
axes[3].set_ylabel('Torque (N m)')
axes[3].legend()
axes[3].grid(True)

# 3. FT_filtered_sensor_ee - FT_1 (첫 3개 컴포넌트) 플롯
# Plot FT_filtered_sensor_ee - FT_1 (first 3 components)
axes[4].plot(FT_filtered_sensor_ee_data.index, FT_filtered_sensor_ee_data[FT_filtered_sensor_ee_cols[0]], label=f'F_x')
axes[4].plot(FT_filtered_sensor_ee_data.index, FT_filtered_sensor_ee_data[FT_filtered_sensor_ee_cols[1]], label=f'F_y')
axes[4].plot(FT_filtered_sensor_ee_data.index, FT_filtered_sensor_ee_data[FT_filtered_sensor_ee_cols[2]], label=f'F_z')
axes[4].set_title('Filtered Force Sensor Data')
axes[4].set_ylabel('Force (N)')
axes[4].legend()
axes[4].grid(True)

# 4. FT_filtered_sensor_ee - FT_2 (다음 3개 컴포넌트) 플롯
# Plot FT_filtered_sensor_ee - FT_2 (next 3 components)
axes[5].plot(FT_filtered_sensor_ee_data.index, FT_filtered_sensor_ee_data[FT_filtered_sensor_ee_cols[3]], label=f'T_x')
axes[5].plot(FT_filtered_sensor_ee_data.index, FT_filtered_sensor_ee_data[FT_filtered_sensor_ee_cols[4]], label=f'T_y')
axes[5].plot(FT_filtered_sensor_ee_data.index, FT_filtered_sensor_ee_data[FT_filtered_sensor_ee_cols[5]], label=f'T_z')
axes[5].set_title('Filtered Torque Sensor Data')
axes[5].set_xlabel('Time Step')
axes[5].set_ylabel('Torque (N m)')
axes[5].legend()
axes[5].grid(True)

plt.tight_layout(rect=[0, 0.03, 1, 0.96]) # 레이아웃 조정
# Adjust layout

# 이미지 저장 경로
# Save image path
save_path = "/home/hyunho_RCI/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/insertion/plot/result/all/1.png"
plt.savefig(save_path)
print(f"Plot saved as '{save_path}'")