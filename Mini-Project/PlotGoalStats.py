import pandas as pd
import matplotlib.pyplot as plt
import os

# === CONFIG ===
RESULTS_DIR = "C:/Users/simon/Desktop/MobileRobotNavigation/Mini-Project/Results/Samuele/03_eps_decay_1500"
CSV_NAME = "metrics.csv"
# === LOAD DATA ===
csv_path = os.path.join(RESULTS_DIR, CSV_NAME)
df = pd.read_csv(csv_path)

# === MAPPING COORDINATE -> LABEL ===
goal_mapping = {
    (-0.7, -0.5): "easy",
    (0.0, 0.5): "medium",
    (1.0, -0.5): "hard",
    (1.7, 0.5): "final"
}

# Creiamo una colonna con la label del goal
df['goal_label'] = list(zip(df['goal_x'], df['goal_y']))
df['goal_label'] = df['goal_label'].map(goal_mapping)


# Assicuriamoci che success sia numerico (0/1)
df['success'] = df['success'].astype(int)

# === GROUP BY GOAL LABEL ===
grouped = df.groupby('goal_label')

stats = grouped.agg(
    total_episodes=('episode', 'count'),
    success_count=('success', 'sum'),
    success_rate=('success', 'mean'),
    avg_steps=('steps', 'mean')
).reset_index()

order = ["easy", "medium", "hard", "final"]
stats = stats.set_index('goal_label').loc[order].reset_index()

# ===============================
# === GRAFICO 1: SUCCESS RATE ===
# ===============================
plt.figure()
plt.bar(stats['goal_label'], stats['success_rate'])
plt.xticks(rotation=45)
plt.xlabel("Goal")
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
plt.xlabel("Goal")
plt.ylabel("Average Steps")
plt.title("Average Steps per Goal")
plt.tight_layout()

steps_plot_path = os.path.join(RESULTS_DIR, "avg_steps_per_goal.png")
plt.savefig(steps_plot_path, dpi=300)
plt.close()

print("Plot salvati in:")
print(success_plot_path)
print(steps_plot_path)
