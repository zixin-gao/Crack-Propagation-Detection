import cv2
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import pandas as pd
from scipy import stats
import tkinter as tk
from tkinter import *
import csv

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
        
    return thresh, 0  # Return 0 instead of None when no crack is found

def display_image_with_zoom_and_roi(img, initial_roi=None):
    """
    Displays a fullscreen, zoomable window of the image at image_path.
    ROI is stored and returned in (x, y, w, h) format (original-image pixels).

    - Mouse wheel: zoom in/out (point under cursor remains fixed).
    - Left click & drag: draw a new ROI 
    - 'k': keep the current ROI and exit (returns it).
    - 'n': clear any existing ROI (and draw anew).
    - Esc: exit without keeping (returns None).

    If initial_roi=(x, y, w, h) is provided, it is drawn on startup.
    """
    cv2.namedWindow("Image Viewer", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty("Image Viewer", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    # -----------------------
    # State for zoom + pan
    # -----------------------
    zoom_level = 1.0
    zoom_center = (img.shape[1] // 2, img.shape[0] // 2)  # (orig_x, orig_y)

    # -----------------------
    # State for ROI in (x,y,w,h)
    # -----------------------
    roi_xywh = None
    if initial_roi is not None:
        x_i, y_i, w_i, h_i = initial_roi
        roi_xywh = (x_i, y_i, w_i, h_i)

    # -----------------------
    # Temp state while drawing
    # -----------------------
    drawing = False
    draw_start = None  # (orig_x, orig_y) where LBUTTONDOWN happened
    draw_end = None    # (orig_x, orig_y) current mouse position while dragging

    def compute_offsets():
        """
        Returns the clamped offsets (x_off_clamped, y_off_clamped) that xpos, ypos in the
        resized image must be shifted by so that we show a w×h patch around zoom_center.

        That is, if the original image is (w, h), and zoom_level scales it to (new_w, new_h),
        we want to pick x_off,y_off so that the rectangle [x_off : x_off + w, y_off : y_off + h]
        is the window we display.

        - un_clamped_x = zoom_center[0]*zoom_level - w/2
        - un_clamped_y = zoom_center[1]*zoom_level - h/2
        - then clamp so that 0 <= x_off <= new_w - w,  0 <= y_off <= new_h - h
        """
        h_img, w_img = img.shape[:2]
        new_w = int(w_img * zoom_level)
        new_h = int(h_img * zoom_level)

        un_x = zoom_center[0] * zoom_level - (w_img / 2)
        un_y = zoom_center[1] * zoom_level - (h_img / 2)

        # clamp:
        x_off = int(round(max(0, min(new_w - w_img, un_x))))
        y_off = int(round(max(0, min(new_h - h_img, un_y))))

        return x_off, y_off, new_w, new_h

    def update_image():
        """
        Resizes + crops the image according to zoom_level & zoom_center, then overlays:
          - the stored ROI (if any and not currently dragging)
          - a rubberband rectangle if drawing==True
        """
        nonlocal zoom_level, zoom_center, roi_xywh, draw_start, draw_end, drawing

        h_img, w_img = img.shape[:2]
        x_off, y_off, new_w, new_h = compute_offsets()

        # Resize full image:
        resized = cv2.resize(img, (new_w, new_h))

        # Crop the w×h window at (x_off, y_off)
        display_patch = resized[y_off:y_off + h_img, x_off:x_off + w_img].copy()

        # --- If there's a stored ROI (and not actively drawing), draw it ---
        if roi_xywh is not None and not drawing:
            x0o, y0o, wo, ho = roi_xywh
            x1o, y1o = x0o + wo, y0o + ho

            # Convert original‐coords → scaled, then subtract offsets:
            x0s = int(round(x0o * zoom_level)) - x_off
            y0s = int(round(y0o * zoom_level)) - y_off
            x1s = int(round(x1o * zoom_level)) - x_off
            y1s = int(round(y1o * zoom_level)) - y_off

            # Clamp into [0..w_img), [0..h_img)
            x0d = max(0, min(w_img - 1, x0s))
            y0d = max(0, min(h_img - 1, y0s))
            x1d = max(0, min(w_img - 1, x1s))
            y1d = max(0, min(h_img - 1, y1s))

            cv2.rectangle(display_patch, (x0d, y0d), (x1d, y1d), (0, 255, 0), 2)

        # --- If we are mid-drawing, overlay the rubberband rectangle ---
        if drawing and draw_start is not None and draw_end is not None:
            x0o, y0o = draw_start
            x1o, y1o = draw_end

            # Determine top-left / bottom-right in orig coords:
            x_min_o, x_max_o = min(x0o, x1o), max(x0o, x1o)
            y_min_o, y_max_o = min(y0o, y1o), max(y0o, y1o)

            # Convert corners to scaled coords, then subtract offsets:
            x0s = int(round(x_min_o * zoom_level)) - x_off
            y0s = int(round(y_min_o * zoom_level)) - y_off
            x1s = int(round(x_max_o * zoom_level)) - x_off
            y1s = int(round(y_max_o * zoom_level)) - y_off

            # Clamp into [0..w_img), [0..h_img)
            x0d = max(0, min(w_img - 1, x0s))
            y0d = max(0, min(h_img - 1, y0s))
            x1d = max(0, min(w_img - 1, x1s))
            y1d = max(0, min(h_img - 1, y1s))

            cv2.rectangle(display_patch, (x0d, y0d), (x1d, y1d), (0, 255, 0), 2)

        cv2.imshow("Image Viewer", display_patch)

    def mouse_callback(event, x, y, flags, param):
        """
        Handles:
        - Mouse wheel: zoom in/out, keeping (x,y) fixed.
        - Left button down: start a new ROI (store draw_start).
        - Mouse move (while drawing): update draw_end.
        - Left button up: finalize ROI (compute roi_xywh).
        """
        nonlocal zoom_level, zoom_center, roi_xywh, drawing, draw_start, draw_end

        h_img, w_img = img.shape[:2]
        x_off, y_off, new_w, new_h = compute_offsets()

        if event == cv2.EVENT_MOUSEWHEEL:
            # Zoom in / out
            if flags > 0:
                zoom_level *= 1.1
            else:
                zoom_level /= 1.1

            # Recompute zoom_center so that the pixel under (x,y) remains under the mouse
            scaled_x = x + x_off
            scaled_y = y + y_off
            orig_x = scaled_x / zoom_level
            orig_y = scaled_y / zoom_level
            zoom_center = (orig_x, orig_y)

            update_image()

        elif event == cv2.EVENT_LBUTTONDOWN:
            # Start drawing a new ROI
            drawing = True
            # Map (x,y) in display_patch → original coords
            scaled_x0 = x + x_off
            scaled_y0 = y + y_off
            orig_x0 = scaled_x0 / zoom_level
            orig_y0 = scaled_y0 / zoom_level
            draw_start = (orig_x0, orig_y0)
            draw_end = (orig_x0, orig_y0)
            update_image()

        elif event == cv2.EVENT_MOUSEMOVE and drawing:
            # Update the rubberband endpoint
            scaled_x1 = x + x_off
            scaled_y1 = y + y_off
            orig_x1 = scaled_x1 / zoom_level
            orig_y1 = scaled_y1 / zoom_level
            draw_end = (orig_x1, orig_y1)
            update_image()

        elif event == cv2.EVENT_LBUTTONUP and drawing:
            drawing = False
            scaled_x1 = x + x_off
            scaled_y1 = y + y_off
            orig_x1 = scaled_x1 / zoom_level
            orig_y1 = scaled_y1 / zoom_level
            draw_end = (orig_x1, orig_y1)

            # Determine integer top-left & bottom-right
            x0i, y0i = map(int, draw_start)
            x1i, y1i = map(int, draw_end)
            x_min, x_max = min(x0i, x1i), max(x0i, x1i)
            y_min, y_max = min(y0i, y1i), max(y0i, y1i)

            # Convert to (x, y, w, h)
            w_roi = x_max - x_min
            h_roi = y_max - y_min
            roi_xywh = (x_min, y_min, w_roi, h_roi)

            # print(f"New ROI (x, y, w, h): {roi_xywh}")
            update_image()

    cv2.setMouseCallback("Image Viewer", mouse_callback)
    update_image()

    final_roi = None
    while True:
        key = cv2.waitKey(20) & 0xFF
        if key == 27:  # Esc: exit without keeping
            break

        elif key == 13: # enter
            # Keep current ROI if it exists
            if roi_xywh is not None:
                final_roi = roi_xywh
                break
            else:
                print("No ROI to keep. Press 'n' to draw one.")

        elif key == ord('n'):
            # Clear any existing ROI to draw a new one
            roi_xywh = None
            drawing = False
            draw_start = None
            draw_end = None
            update_image()
            print("Cleared old ROI. Left-click & drag to draw a new one.")

    cv2.destroyAllWindows()
    return final_roi

def input_skip_frame():
    main_window = Tk()

    def close():
        main_window.destroy()

    frame_button_var = IntVar()
    options = [1, 3, 5, 10, 20, 30, 50, 100, 300, 500, 1000]

    for option in options:
        Radiobutton(
            main_window, 
            text=f"{option} Frame(s)", 
            variable=frame_button_var, 
            value=option, 
            pady='5',
            padx='5',
            font=("Times New Roman", 12),
            command=close
        ).pack(anchor=W)

    main_window.mainloop()

    return frame_button_var.get()

def process_image_sequence(image_folder, reference_pixels, reference_mm, frame_rate, threshold=None):
    """Processes images one by one, with debugger"""
    image_files = sorted([f for f in os.listdir(image_folder)])
    crack_lengths = []
    times = []
    prev_roi = None
    prev_frame = 0
    N = 0

    for i, img_file in enumerate(tqdm(image_files)):
        img = cv2.imread(os.path.join(image_folder, img_file))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)

        # Update ROI periodically or when requested
        if i == 0 or (i - prev_frame) >= N or i == len(image_files) - 1:
            x, y, w, h = display_image_with_zoom_and_roi(thresh, prev_roi)
            prev_roi = (x, y, w, h)
        else:
            x, y, w, h = prev_roi
            
        img = img[y:y+h, x:x+w]
        thresh, tip_x = detect_crack_tip(img, threshold)
        
        # Always record data (tip_x will be 0 if no crack found)
        mm_length = (tip_x / reference_pixels) * reference_mm
        crack_lengths.append(mm_length)
        times.append(i * (1/frame_rate))

        if i == 0 or i == prev_frame + N:
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
            fig = plt.figure(figsize=(10, 3.5))
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

            N = input_skip_frame()
            prev_frame = i

    return times, crack_lengths

# def remove_extreme_outliers(times, lengths):
#     """removes the outliers that are outside of 3 standard deviation"""

#     z = np.abs(stats.zscore(lengths))
#     threshold_z = 1
#     outlier_indices = np.where(z > threshold_z)[0]
#     # print(np.where(z > threshold_z))
#     # print(outlier_indices)
#     # print(lengths)

#     no_outlier_lengths = []
#     no_outlier_times = []
    
#     for i in range(len(lengths)):
#         if i not in outlier_indices:
#             no_outlier_lengths.append(lengths[i])
#             no_outlier_times.append(times[i])

#     return no_outlier_times, no_outlier_lengths

def plot_results(times, lengths):
    """plot data points and display on a graph"""

    # filtered_times, filtered_lengths = remove_extreme_outliers(times, lengths)
    plt.figure(figsize=(12, 6))
    plt.scatter(times, lengths, s=20, c='blue', marker='x', alpha=0.5)
    plt.xlabel("time (s)")
    plt.ylabel("crack propagation (mm)")
    plt.title("Crack Propagation Over Time", fontweight='bold')
    plt.grid(alpha=0.3)
    plt.savefig("./Data/Plot.png")

    thismanager = plt.get_current_fig_manager()
    thismanager.window.wm_geometry("+400+330")
    plt.show()

def read_from_csv(csv_path):
    times = []
    lengths = []

    with open(csv_path, 'r') as csvfile:
        lines = csv.reader(csvfile, delimiter=',')
        next(lines)
        for line in lines:
            times.append(float(line[0]))
            lengths.append(float(line[1]))
    
    return times, lengths

if __name__ == "__main__":

    image_folder = r"E:\2025 summe research\3 - tensile coupon\D5528 June 9 Test"     
    reference_pixels = 122                                             # reference (px) 
    reference_mm = 2                                                    # reference (mm)
    threshold = 35                                                    # RGB threshold to binary image
    frame_rate = 2                                                      # frame rate (Hz)

    times, lengths = process_image_sequence(
        image_folder,
        reference_pixels,
        reference_mm,
        
        frame_rate,
        threshold
    )

    # Create a complete timeline with all frames
    all_times = [i * (1/frame_rate) for i in range(len(os.listdir(image_folder)))]
    all_lengths = [0.0] * len(all_times)
    
    # Fill in the measured values
    for t, l in zip(times, lengths):
        idx = int(t * frame_rate)
        if idx < len(all_lengths):
            all_lengths[idx] = l
    
    # Save the complete data
    complete_data = pd.DataFrame({
        'time elapsed (s)': all_times,
        'crack length (mm)': all_lengths
    })

    plot_results(all_times, all_lengths)

    complete_data.to_csv('./Data/complete_data.csv', float_format='%.3f', index=False)
