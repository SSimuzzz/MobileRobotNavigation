import pandas as pd
import matplotlib.pyplot as plt

# Legge il file CSV
FILE_PATH = "C:/Users/simon/Desktop/MobileRobotNavigation/Mini-Project/Results/Samuele/01/metrics.csv"
df = pd.read_csv(FILE_PATH)

# Plot della colonna epsilon
plt.plot(df["epsilon"])
plt.xlabel("Episodio")
plt.ylabel("epsilon")
plt.title("Plot di epsilon")
plt.grid(True)
plt.show()
