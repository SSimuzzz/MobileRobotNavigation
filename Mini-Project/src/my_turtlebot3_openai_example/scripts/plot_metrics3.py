#!/usr/bin/env python3

import os
import csv
import numpy as np
import matplotlib.pyplot as plt

# ==========================================================
# CONFIGURAZIONE
# ==========================================================
#RESULTS_DIR = "../training_results" 
RESULTS_DIR = "C:/Users/s.simonitti/Desktop/Nuova cartella/minipro"
CSV_NAME = "metrics.csv"
WINDOW = 50  # Finestra per medie mobili

csv_path = os.path.join(RESULTS_DIR, CSV_NAME)

def rolling_average(data, window):
    """Calcola la media mobile semplice usando solo Numpy."""
    if len(data) == 0:
        return np.array([])
    data_arr = np.array(data, dtype=float)
    out = np.zeros_like(data_arr)
    for i in range(len(data_arr)):
        start = max(0, i - window + 1)
        out[i] = np.mean(data_arr[start:i+1])
    return out

def generate_plots():
    if not os.path.exists(csv_path):
        print(f"ERRORE: File {csv_path} non trovato.")
        return

    # 1. CARICAMENTO DATI (Manuale con modulo CSV)
    data = []
    try:
        with open(csv_path, mode='r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Convertiamo i valori e puliamo i nomi delle colonne da spazi
                clean_row = {k.strip(): float(v) for k, v in row.items()}
                data.append(clean_row)
    except Exception as e:
        print(f"Errore durante la lettura del CSV: {e}")
        return

    if not data:
        print("Il file CSV è vuoto.")
        return

    # Estrazione colonne principali
    episodes = [d['episode'] for d in data]
    rewards = [d['reward'] for d in data]
    successes = [d['success'] for d in data]
    collisions = [d['collision'] for d in data]

    # ==========================================================
    # GRAFICO 1: TOTAL REWARD (Immagine Distinta)
    # ==========================================================
    plt.figure(figsize=(12, 6))
    reward_avg = rolling_average(rewards, WINDOW)
    
    plt.plot(episodes, rewards, color='gray', alpha=0.3, label='Raw Reward')
    plt.plot(episodes, reward_avg, color='blue', linewidth=2, label=f'Trend (MA {WINDOW})')
    
    plt.title('Total Reward per Episode', fontsize=14, fontweight='bold')
    plt.xlabel('Episode')
    plt.ylabel('Cumulative Reward')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig(os.path.join(RESULTS_DIR, "plot_01_total_reward.png"))
    print("Salvato: plot_01_total_reward.png")

    # ==========================================================
    # GRAFICO 2: PERFORMANCE (SUCCESS & COLLISION RATE)
    # ==========================================================
    plt.figure(figsize=(12, 6))
    success_rate = rolling_average(successes, WINDOW) * 100
    collision_rate = rolling_average(collisions, WINDOW) * 100
    
    plt.plot(episodes, success_rate, color='green', linewidth=2, label='Success Rate %')
    plt.plot(episodes, collision_rate, color='red', linewidth=2, label='Collision Rate %')
    plt.fill_between(episodes, success_rate, color='green', alpha=0.1)
    plt.fill_between(episodes, collision_rate, color='red', alpha=0.1)
    
    plt.title(f'Success vs Collision Rate (MA {WINDOW})', fontsize=14, fontweight='bold')
    plt.xlabel('Episode')
    plt.ylabel('Percentage (%)')
    plt.ylim(-5, 105)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig(os.path.join(RESULTS_DIR, "plot_02_performance.png"))
    print("Salvato: plot_02_performance.png")

    # ==========================================================
    # GRAFICO 3: TIME TO TARGET (SOLO SUCCESSI)
    # ==========================================================
    plt.figure(figsize=(12, 6))
    succ_episodes = [d['episode'] for d in data if d['success'] == 1]
    succ_steps = [d['steps'] for d in data if d['success'] == 1]

    if succ_episodes:
        plt.scatter(succ_episodes, succ_steps, color='darkblue', s=12, alpha=0.4, label='Steps (Success)')
        trend_steps = rolling_average(succ_steps, 15)
        plt.plot(succ_episodes, trend_steps, color='cyan', linewidth=2, label='Efficiency Trend')
        
        plt.title('Steps to Reach Goal (Successful Episodes)', fontsize=14, fontweight='bold')
        plt.xlabel('Episode')
        plt.ylabel('Steps')
        plt.grid(True, alpha=0.3)
        plt.legend()
    else:
        plt.text(0.5, 0.5, "Nessun successo registrato", ha='center', va='center')

    plt.savefig(os.path.join(RESULTS_DIR, "plot_03_time_to_target.png"))
    print("Salvato: plot_03_time_to_target.png")

    # ==========================================================
    # GRAFICO 4: REWARD COMPONENTS BREAKDOWN
    # ==========================================================
    components = ['r_progress', 'r_time', 'r_yaw', 'r_collision_avoid', 'r_terminal']
    available = [c for c in components if c in data[0]]
    
    if available:
        fig, axes = plt.subplots(len(available), 1, figsize=(12, 12), sharex=True)
        if len(available) == 1: axes = [axes]
        
        for i, col in enumerate(available):
            comp_vals = [d[col] for d in data]
            comp_avg = rolling_average(comp_vals, WINDOW)
            axes[i].plot(episodes, comp_vals, color='gray', alpha=0.2)
            axes[i].plot(episodes, comp_avg, label=col, linewidth=2)
            axes[i].set_ylabel('Value')
            axes[i].legend(loc='upper left')
            axes[i].grid(True, alpha=0.2)
        
        plt.xlabel('Episode')
        plt.suptitle('Factorized Reward Components', fontsize=16, fontweight='bold')
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(os.path.join(RESULTS_DIR, "plot_04_reward_breakdown.png"))
        print("Salvato: plot_04_reward_breakdown.png")

    plt.show()

if __name__ == "__main__":
    generate_plots()