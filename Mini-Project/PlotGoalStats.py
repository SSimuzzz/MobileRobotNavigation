import pandas as pd
import matplotlib.pyplot as plt
import os

# === CONFIG ===
RESULTS_DIR = "C:/Users/simon/Desktop/MobileRobotNavigation/Mini-Project/Results/Samuele/03_eps_decay_1500"
CSV_NAME = "metrics.csv"
# === LOAD DATA ===
csv_path = os.path.join(RESULTS_DIR, CSV_NAME)
df = pd.read_csv(csv_path)

# Assicuriamoci che success sia numerico (0/1)
df['success'] = df['success'].astype(int)

# === GROUP BY GOAL COORDINATES ===
grouped = df.groupby(['goal_x', 'goal_y'])

# Calcolo statistiche
stats = grouped.agg(
    total_episodes=('episode', 'count'),
    success_count=('success', 'sum'),
    success_rate=('success', 'mean'),   # media di 0/1 = success rate
    avg_steps=('steps', 'mean')
).reset_index()

# Creo una label stringa per ogni goal
stats['goal_label'] = stats.apply(
    lambda row: f"({row['goal_x']}, {row['goal_y']})",
    axis=1
)

# ===============================
# === GRAFICO 1: SUCCESS RATE ===
# ===============================
plt.figure()
plt.bar(stats['goal_label'], stats['success_rate'])
plt.xticks(rotation=45)
plt.xlabel("Goal (x, y)")
plt.ylabel("Success Rate")
plt.title("Success Rate per Goal")
plt.tight_layout()

success_plot_path = os.path.join(RESULTS_DIR, "success_rate_per_goal.png")
plt.savefig(success_plot_path, dpi=300)
plt.close()

# ===============================
# === GRAFICO 2: AVG STEPS ===
# ===============================
plt.figure()
plt.bar(stats['goal_label'], stats['avg_steps'])
plt.xticks(rotation=45)
plt.xlabel("Goal (x, y)")
plt.ylabel("Average Steps")
plt.title("Average Steps per Goal")
plt.tight_layout()

steps_plot_path = os.path.join(RESULTS_DIR, "avg_steps_per_goal.png")
plt.savefig(steps_plot_path, dpi=300)
plt.close()

print("Plot salvati in:")
print(success_plot_path)
print(steps_plot_path)
