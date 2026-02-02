import numpy as np
import matplotlib.pyplot as plt
import math

# -----------------------------
# PARAMETRI
# -----------------------------
w_progress = 40.0
w_yaw = 1.0
w_collision = 2.0
r_time = -0.05
max_steps_per_episode = 800

# Progress reward
DELTA_DIST_MAX = 6  # [m/step] <-- PARAMETRO CHIAVE

LIDAR_MAX_RANGE = 10.0

N_BEAMS = 181
FOV = np.pi  # [-90°, +90°]

MAX_WEIGHT = 10.0
POWER = 6
RANGE_VALID_MAX = 0.5
ROBOT_RADIUS = 0.25
DECAY_K = 3.0

# -----------------------------
# Funzioni
# -----------------------------
def compute_directional_weights(relative_angles, max_weight=MAX_WEIGHT, power=POWER):
    raw_weights = (np.cos(relative_angles)) ** power + 0.1
    scaled_weights = raw_weights * (max_weight / np.max(raw_weights))
    normalized_weights = scaled_weights / np.sum(scaled_weights)
    return normalized_weights

def compute_weighted_obstacle_reward_as_is(front_ranges, front_angles):
    # identica alla tua logica
    front_ranges = np.array(front_ranges, dtype=float)
    front_angles = np.array(front_angles, dtype=float)

    valid_mask = front_ranges <= RANGE_VALID_MAX
    if not np.any(valid_mask):
        return 0.0

    r = front_ranges[valid_mask]
    a = front_angles[valid_mask]

    relative_angles = np.unwrap(a)
    relative_angles[relative_angles > np.pi] -= 2 * np.pi

    weights = compute_directional_weights(relative_angles, max_weight=MAX_WEIGHT)

    safe_dists = np.clip(r - ROBOT_RADIUS, 1e-2, 3.5)
    decay = np.exp(-DECAY_K * safe_dists)
    weighted_decay = np.dot(weights, decay)

    return - (1.0 + 4.0 * weighted_decay)

def compute_weighted_obstacle_reward_analysis(front_ranges, front_angles):
    """
    Versione SOLO per analisi:
    - nessun filtro a 0.5m
    - pesi calcolati su tutti gli angoli
    - decay su tutti i raggi (quelli lontani contribuiscono poco perché exp(-k*dist))
    """
    r = np.array(front_ranges, dtype=float)
    a = np.array(front_angles, dtype=float)

    relative_angles = np.unwrap(a)
    relative_angles[relative_angles > np.pi] -= 2 * np.pi
    weights = compute_directional_weights(relative_angles, max_weight=MAX_WEIGHT)

    safe_dists = np.clip(r - ROBOT_RADIUS, 1e-2, 3.5)
    decay = np.exp(-DECAY_K * safe_dists)
    weighted_decay = np.dot(weights, decay)

    return - (1.0 + 4.0 * weighted_decay)

def yaw_reward(goal_angle_rad):
    return w_yaw * (1.0 - (2.0 * np.abs(goal_angle_rad) / np.pi))

def progress_reward(delta_dist, delta_dist_max=DELTA_DIST_MAX):
    delta_clipped = np.clip(delta_dist,
                             -delta_dist_max,
                             +delta_dist_max)
    return w_progress * delta_clipped

def time_reward(steps):
    return r_time * steps

def scan_with_single_obstacle(front_angles, obstacle_angle, obstacle_distance,
                              lidar_max_range=LIDAR_MAX_RANGE, beam_spread=5):
    ranges = np.full_like(front_angles, fill_value=lidar_max_range, dtype=float)
    idx = int(np.argmin(np.abs(front_angles - obstacle_angle)))
    lo = max(0, idx - beam_spread)
    hi = min(len(ranges), idx + beam_spread + 1)
    ranges[lo:hi] = obstacle_distance
    return ranges

# -----------------------------
# Domini
# -----------------------------
delta_d = np.linspace( -DELTA_DIST_MAX,
                       DELTA_DIST_MAX,
                       400)

steps = np.arange(0, max_steps_per_episode + 1)

angles_deg = np.linspace(-180, 180, 600)
angles_rad = np.deg2rad(angles_deg)

front_angles = np.linspace(-FOV/2, FOV/2, N_BEAMS)
obstacle_distances = np.linspace(0.05, LIDAR_MAX_RANGE, 400)

obstacle_cases = [
    ("0° frontale", np.deg2rad(0.0)),
    ("45°", np.deg2rad(45.0)),
    ("90°", np.deg2rad(90.0)),
]

# -----------------------------
# Curve
# -----------------------------
r_prog = progress_reward(delta_d)
r_yaw = yaw_reward(angles_rad)
r_t = time_reward(steps)

collision_as_is = {}
collision_analysis = {}

for label, a in obstacle_cases:
    vals1, vals2 = [], []
    for d_obs in obstacle_distances:
        scan = scan_with_single_obstacle(front_angles, a, d_obs, beam_spread=5)
        vals1.append(w_collision * compute_weighted_obstacle_reward_as_is(scan, front_angles))
        vals2.append(w_collision * compute_weighted_obstacle_reward_analysis(scan, front_angles))
    collision_as_is[label] = np.array(vals1)
    collision_analysis[label] = np.array(vals2)

# -----------------------------
# Time
# -----------------------------
steps = np.arange(0, max_steps_per_episode + 1)
r_t = time_reward(steps)

# -----------------------------
# Plot
# -----------------------------
fig = plt.figure(figsize=(13, 9))

ax1 = plt.subplot(2, 2, 1)
ax1.plot(delta_d, r_prog)
ax1.axvline(0.0, linewidth=1)
ax1.set_title("Progress reward vs Δd")
ax1.set_xlabel("Δd = prev_dist - dist  [m/step]")
ax1.set_ylabel("reward")

ax2 = plt.subplot(2, 2, 2)
ax2.plot(angles_deg, r_yaw)
ax2.axvline(0.0, linewidth=1)
ax2.set_title("Yaw reward (asse in gradi)")
ax2.set_xlabel("goal_angle [deg]")
ax2.set_ylabel("reward")

"""ax3 = plt.subplot(2, 2, 3)
for label, curve in collision_as_is.items():
    ax3.plot(obstacle_distances, curve, label=label)
ax3.set_title("Collision (AS-IS: filtro range<=0.5m)")
ax3.set_xlabel("obstacale distance [m]")
ax3.set_ylabel("reward")
ax3.legend()"""

ax3 = plt.subplot(2, 2, 3)
for label, curve in collision_analysis.items():
    ax3.plot(obstacle_distances, curve, label=label)
ax3.set_title("Collision (ANALYSIS: senza filtro 0.5m)")
ax3.set_xlabel("obstacale distance [m]")
ax3.set_ylabel("reward")
ax3.legend()

ax4 = plt.subplot(2, 2, 4)
ax4.plot(steps, r_t)
ax4.set_title("Time reward")
ax4.set_xlabel("steps")
ax4.set_ylabel("reward")

plt.tight_layout()
plt.show()
