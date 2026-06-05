import re
import pandas as pd
import numpy as np

def clean_license_plate(text):
    if not text:
        return ""

    cleaned = re.sub(r'[^A-Z0-9]', '', text.upper())
    return cleaned[:15]

def character_mapping(char):
    """Map similar looking characters to correct ones"""
    mapping = {
        'O': '0', '0': '0',
        'I': '1', '1': '1', 'L': '1',
        'S': '5', '5': '5',
        'Z': '2', '2': '2',
        'B': '8', '8': '8',
        'G': '6', '6': '6'
    }
    return mapping.get(char, char)

def validate_plate(plate_text):
    """Validate and score license plate based on common formats"""
    if not plate_text or len(plate_text) < 3:
        return plate_text, 0.0
    
    # Apply character mapping
    mapped_plate = ''.join(character_mapping(c) for c in plate_text)
    
    # Common patterns that increase confidence
    patterns = [
        r'^[A-Z]{1,3}[0-9]{1,4}[A-Z]{0,2}$',  # US/UK style
        r'^[0-9]{1,3}[A-Z]{1,3}$',            # Numeric first
        r'^[A-Z]{2,3}-[0-9]{3,4}$',           # With dash
    ]
    
    confidence = 0.5  # Base confidence
    
    # Length-based scoring
    length = len(mapped_plate)
    if 5 <= length <= 10:
        confidence += 0.3
    elif length > 10:
        confidence += 0.1
    
    # Pattern matching
    for pattern in patterns:
        if re.match(pattern, mapped_plate):
            confidence += 0.4
            break
    
    # All numeric or all letters reduces confidence
    if mapped_plate.isdigit() or mapped_plate.isalpha():
        confidence *= 0.7
    
    # Has both letters and numbers increases confidence
    if re.search(r'[A-Z]', mapped_plate) and re.search(r'[0-9]', mapped_plate):
        confidence += 0.2
    
    return mapped_plate, min(confidence, 1.0)

def interpolate_missing_bboxes(results_df, max_gap=5):
    """Interpolate missing bounding boxes for smoother tracking"""
    interpolated_results = []
    
    for car_id in results_df['car_id'].unique():
        car_data = results_df[results_df['car_id'] == car_id].sort_values('frame_number')
        frames = car_data['frame_number'].values
        
        for i in range(len(frames)):
            current_frame = frames[i]
            interpolated_results.append(car_data.iloc[i].to_dict())
            
            # Interpolate forward for missing frames
            next_frame = frames[i + 1] if i + 1 < len(frames) else None
            if next_frame and next_frame - current_frame <= max_gap:
                for gap_frame in range(current_frame + 1, next_frame):
                    interp_data = car_data.iloc[i].copy()
                    interp_data['frame_number'] = gap_frame
                    interp_data['license_number_score'] *= 0.8  # Reduce confidence for interpolated
                    interpolated_results.append(interp_data)
    
    return pd.DataFrame(interpolated_results)
