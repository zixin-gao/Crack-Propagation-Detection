import cv2
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import pandas as pd
from scipy import stats

def detect_crack_tip(img, threshold):
    """measures the pixel which the crack tip is at"""

    # converting to binary image
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)

    # find right most black pixel
    row, col = thresh.shape

    for x in range(col-1, -1, -1):
        if 0 in thresh[:, x]:
            return thresh, x    # this is the col that contains the black pixel
        
    return thresh, None

def process_image_sequence(image_folder, reference_pixels, reference_mm, frame_rate, roi=None, threshold=None):
    """Processes images one by one, with debugger"""
    
    image_files = sorted([f for f in os.listdir(image_folder)])
    crack_lengths = []
    times = []

    # loop through image, add in progress bar
    for i, img_file in enumerate(tqdm(image_files)):
        img = cv2.imread(os.path.join(image_folder, img_file))

        if img is None:
            continue

        if roi:
            x, y, w, h = roi
            img = img[y:y+h, x:x+w]

        thresh, tip_x = detect_crack_tip(img, threshold)

        if tip_x is not None:
            mm_length = (tip_x / reference_pixels) * reference_mm
        
            # record time and tip length
            crack_lengths.append(mm_length)
            times.append(i * (1/frame_rate))
            
            N = 100     # debugging every N frames - change when needed

            if i % N == 0:
                # drawing line at the crack tip
                img_cpy = img.copy()
                thresh_cpy = cv2.cvtColor(thresh.copy(), cv2.COLOR_GRAY2BGR)

                line_start = (tip_x, 0)
                line_end = (tip_x, img.shape[0])
                line_color = (0, 0, 255)  # BGR
                line_thickness = 2

                cv2.line(img_cpy, line_start, line_end, line_color, line_thickness)
                cv2.line(thresh_cpy, line_start, line_end, line_color, line_thickness)

                # Figure plot
                fig = plt.figure(figsize=(8, 3.5))
                fig.suptitle(f"Frame {i}: Crack at {tip_x}px, Crack Length at {mm_length:.2f}mm", fontweight='bold')

                plt.subplot(2, 1, 1)
                plt.imshow(cv2.cvtColor(thresh_cpy, cv2.COLOR_BGR2RGB))
                plt.title(f"Binary Threshold Image", fontweight='bold')
                plt.xlabel("Pixel Count")

                plt.subplot(2, 1, 2)
                plt.imshow(cv2.cvtColor(img_cpy, cv2.COLOR_BGR2RGB))
                plt.title(f"Original Image", fontweight='bold')
                plt.xlabel("Pixel Count")

                # make plots always show up at the same position
                thismanager = plt.get_current_fig_manager()
                thismanager.window.wm_geometry("+700+430")
                plt.show()

    return times, crack_lengths

def remove_extreme_outliers(times, lengths):
    """removes the outliers that are outside of 3 standard deviation"""

    z = np.abs(stats.zscore(lengths))
    threshold_z = 1
    outlier_indices = np.where(z > threshold_z)[0]
    # print(np.where(z > threshold_z))
    # print(outlier_indices)
    # print(lengths)

    no_outlier_lengths = []
    no_outlier_times = []
    
    for i in range(len(lengths)):
        if i not in outlier_indices:
            no_outlier_lengths.append(lengths[i])
            no_outlier_times.append(times[i])

    return no_outlier_times, no_outlier_lengths

def plot_results(times, lengths):
    """plot data points and display on a graph"""

    filtered_times, filtered_lengths = remove_extreme_outliers(times, lengths)
    plt.figure(figsize=(12, 6))
    plt.scatter(filtered_times, filtered_lengths, s=20, c='blue', marker='x', alpha=0.5)
    plt.xlabel("time (s)")
    plt.ylabel("crack propagation (mm)")
    plt.title("Crack Propagation Over Time", fontweight='bold')
    plt.grid(alpha=0.3)

    thismanager = plt.get_current_fig_manager()
    thismanager.window.wm_geometry("+400+330")
    plt.show()

if __name__ == "__main__":

    image_folder = "../D5528 May 29 Test Filtered"     # path to folder
    reference_pixels = 140                             # reference (px)
    reference_mm = 2                                   # reference (mm)
    roi = (1200, 350, 1500, 80)                        # input format: (x, y, w, h)
    threshold = 53                                     # RGB threshold to binary image
    frame_rate = 1                                     # frame rate (Hz)
    
    times, lengths = process_image_sequence(
        image_folder,
        reference_pixels,
        reference_mm,
        frame_rate,
        roi,
        threshold
    )

    plot_results(times, lengths)

    # Save to CSV
    raw_data = pd.DataFrame({'time elapsed (s)': times, 'crack length (mm)': lengths})
    raw_data.to_csv('./Data/raw_data.csv', float_format='%.3f', index=False)

    processed_data = pd.DataFrame({'time elapsed (s)': times, 'crack length (mm)': lengths})
    processed_data.to_csv('./Data/processed_data.csv', float_format='%.3f', index=False)