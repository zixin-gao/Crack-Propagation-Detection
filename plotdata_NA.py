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
    plt.title("June 17 2025, PLA-4", fontweight='bold')
    plt.grid(alpha=0.3)
    plt.savefig("./Data/Plot.png")

    thismanager = plt.get_current_fig_manager()
    thismanager.window.wm_geometry("+400+330")
    plt.show()

def filter_NA(csv_filename):
    df = pd.read_csv(csv_filename)
    df_cleaned = df.dropna()
    times = df_cleaned["time elapsed (s)"]
    lengths = df_cleaned["crack length (mm)"]
    
    return times, lengths

if __name__ == '__main__':
    csv_filename = "./Data/complete_data.csv"
    times, lengths = filter_NA(csv_filename)
    plot_results(times, lengths)