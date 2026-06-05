import cv2
import pandas as pd
import numpy as np
from pathlib import Path
from util import interpolate_missing_bboxes

def draw_results_on_video(input_video, results_csv, output_video):
    """Visualize ANPR results on video with interpolation"""
    
    # Load results and interpolate
    df = pd.read_csv(results_csv)
    interpolated_df = interpolate_missing_bboxes(df)
    
    # Create frame lookup dictionary
    frame_data = {}
    for _, row in interpolated_df.iterrows():
        frame_num = int(row['frame_number'])
        if frame_num not in frame_data:
            frame_data[frame_num] = []
        frame_data[frame_num].append(row)
    
    # Open video
    cap = cv2.VideoCapture(input_video)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    # --- CHANGE 1: Hardcode the width/height to match your main.py ---
    width = 1280 
    height = 720
    
    # Setup video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))
    
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # --- CHANGE 2: Resize the frame before drawing anything ---
        frame = cv2.resize(frame, (1280, 720)) 
        
        frame_count += 1
        
        # Get detections for this frame
        # (Note: frame_count logic stays the same)
        if frame_count in frame_data:
            for detection in frame_data[frame_count]:
                # Draw car bbox
                car_bbox_str = detection['car_bbox'].replace('[', '').replace(']', '').replace(',', ' ').strip()
                car_bbox = [int(float(x)) for x in car_bbox_str.split()]
                cv2.rectangle(frame, (car_bbox[0], car_bbox[1]), 
                            (car_bbox[2], car_bbox[3]), (0, 255, 0), 2)
                cv2.putText(frame, f'ID: {int(detection["car_id"])}', 
                           (car_bbox[0], car_bbox[1]-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                # Draw plate bbox
                plate_bbox_str = detection['license_plate_bbox'].replace('[', '').replace(']', '').replace(',', ' ').strip()
                plate_bbox = [int(float(x)) for x in plate_bbox_str.split()] 
                plate_color = (0, 0, 255) if detection['license_number_score'] > 0.5 else (0, 255, 255)
                cv2.rectangle(frame, (int(plate_bbox[0]), int(plate_bbox[1])), 
                            (int(plate_bbox[2]), int(plate_bbox[3])), plate_color, 2)
                
                # Draw license plate text
                # --- CHANGE 3: Add a small check for NaN license numbers ---
                l_num = str(detection['license_number']) if pd.notnull(detection['license_number']) else "UNKNOWN"
                plate_text = f"{l_num} ({detection['license_number_score']:.2f})"
                
                text_size = cv2.getTextSize(plate_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                cv2.rectangle(frame, (int(plate_bbox[0]), int(plate_bbox[1])-text_size[1]-10),
                            (int(plate_bbox[0])+text_size[0], int(plate_bbox[1])), plate_color, -1)
                cv2.putText(frame, plate_text, (int(plate_bbox[0]), int(plate_bbox[1])-5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        # Frame counter
        cv2.putText(frame, f'Frame: {frame_count}', (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        out.write(frame)
        
        if frame_count % 30 == 0:
            print(f"Visualizing frame {frame_count}")
    
    cap.release()
    out.release()
    print(f"Visualization saved to {output_video}")

# ... (rest of the main() function remains the same)
def main():
    # File paths - ensure these match your folder structure exactly
    input_video = 'input_video.mp4'  
    results_csv = 'test_interpolated.csv' # Uses the smoothed data from add_missing_data.py
    output_video = 'output_processed.mp4'
    
    # Check if the required files exist before starting
    video_path = Path(input_video)
    csv_path = Path(results_csv)
    
    if csv_path.exists() and video_path.exists():
        print(f"Starting visualization on: {input_video}")
        print(f"Using data from: {results_csv}")
        
        try:
            draw_results_on_video(input_video, results_csv, output_video)
            print("\n--- Visualization complete! ---")
            print(f"Final video saved as: {output_video}")
        except Exception as e:
            print(f"An error occurred during visualization: {e}")
    else:
        # Debugging message to help you find missing files
        if not video_path.exists():
            print(f"Error: Video file '{input_video}' not found in O:\\number Plate")
        if not csv_path.exists():
            print(f"Error: CSV file '{results_csv}' not found. Did you run add_missing_data.py first?")

if __name__ == "__main__":
    main()