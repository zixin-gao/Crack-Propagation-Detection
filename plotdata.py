import matplotlib.pyplot as plt
import csv
import pandas as pd

def plot_results(times, lengths):
    """plot data points and display on a graph"""

    # filtered_times, filtered_lengths = remove_extreme_outliers(times, lengths)
    plt.figure(figsize=(12, 6))
    plt.scatter(times, lengths, s=20, c='blue', marker='x', alpha=0.5)
    plt.xlabel("time (s)")
    plt.ylabel("crack propagation (mm)")
    plt.title("June 9 2025, PLA-1", fontweight='bold')
    plt.grid(alpha=0.3)
    plt.savefig("./Data/Plot.png")

    thismanager = plt.get_current_fig_manager()
    thismanager.window.wm_geometry("+400+330")
    plt.show()

df = pd.read_csv("./Data/complete_data.csv")
times = df["time elapsed (s)"]
lengths = df["crack length (mm)"]
plot_results(times, lengths)

