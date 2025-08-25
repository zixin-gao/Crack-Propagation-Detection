import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import pandas as pd 

def detect_crack_tip(img, threshold):
    """Finds the rightmost black pixel (crack tip)"""
    # Convert to grayscale and threshold (cracks=black, background=white)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    
    # Find rightmost black pixel (column-wise scan from right to left)
    height, width = thresh.shape
    for x in range(width-1, -1, -1):  # Right-to-left scan
        if 0 in thresh[:, x]:  # Check if column contains black pixels
            return thresh, x  # Return thresholded image and x-position
    
    return thresh, None  # No crack found

def process_image_sequence(image_folder, reference_pixels, roi=None, threshold=None):
    image_files = sorted([f for f in os.listdir(image_folder) if f.endswith(('.png', '.jpg', '.jpeg'))])
    
    crack_lengths = []
    times = []
    
    for i, img_file in enumerate(tqdm(image_files)):
        img = cv2.imread(os.path.join(image_folder, img_file))
        if img is None:
            continue
            
        if roi:
            x, y, w, h = roi
            img = img[y:y+h, x:x+w]
        
        thresh, tip_x = detect_crack_tip(img, threshold)
        
        if tip_x is not None:
            mm_length = (tip_x / reference_pixels) * 2  # Convert to mm
            
            crack_lengths.append(mm_length)
            times.append(i * 0.1)  # 10Hz = 0.1s per frame
            
            # Visualization every N frames
            if i % 50 == 0:
                display_img = img.copy()
                cv2.line(display_img, (tip_x, 0), (tip_x, img.shape[0]), (0, 0, 255), 2)
                
                plt.figure(figsize=(12, 4))
                plt.subplot(121), plt.imshow(thresh, cmap='gray')
                plt.title("Thresholded Image")
                plt.subplot(122), plt.imshow(cv2.cvtColor(display_img, cv2.COLOR_BGR2RGB))
                plt.title(f"Crack Tip at X={tip_x}px ({mm_length:.2f}mm)")
                plt.show()
    
    return times, crack_lengths

def remove_extreme_outliers(times, lengths):
    """Remove only extreme outliers using modified Z-score method"""
    lengths = np.array(lengths)
    times = np.array(times)
    
    # Calculate modified Z-scores (robust to small datasets)
    median = np.median(lengths)
    mad = np.median(np.abs(lengths - median))  # Median Absolute Deviation
    modified_z = 0.6745 * (lengths - median) / mad  # Scaled MAD
    
    # Keep only data within 3.5 modified Z-scores (~3mm for typical cracks)
    mask = np.abs(modified_z) < 3.5
    return times[mask], lengths[mask]

def plot_results(times, lengths):
    plt.figure(figsize=(12, 6))
    
    # Remove only extreme outliers
    clean_times, clean_lengths = remove_extreme_outliers(times, lengths)
    
    # Plot all remaining points (including moderate variations)
    plt.scatter(clean_times, clean_lengths, s=20, color='blue', alpha=0.6, 
               label='Crack Length')
    
    # Trend line (using cleaned data)
    if len(clean_times) > 1:
        z = np.polyfit(clean_times, clean_lengths, 1)
        p = np.poly1d(z)
        plt.plot(clean_times, p(clean_times), 'r-', linewidth=2, label='Trend')
    
    plt.xlabel('Time (seconds)')
    plt.ylabel('Crack Length (mm)')
    plt.title('Crack Propagation Over Time')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    image_folder = "./May29Test"
    reference_pixels = 140          # 2mm reference equivalent pixel width
    roi = (1200, 350, 1500, 80)     # ROI to box the crack
    threshold = 53                  # to adjust the black and white scale
    
    times, lengths = process_image_sequence(
        image_folder=image_folder,
        reference_pixels=reference_pixels,
        roi=roi,
        threshold=threshold
    )
    
    # Save results
    df = pd.DataFrame({'Time(s)': times, 'Length(mm)': lengths})
    df.to_csv('crack_growth_boundingbox.csv', index=False)
    
    # Plot results
    plot_results(times, lengths)