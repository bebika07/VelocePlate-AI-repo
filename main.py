import argparse
import os
from pathlib import Path

import cv2
import easyocr
import numpy as np
import pandas as pd
import torch
from sort import Sort
from ultralytics import YOLO

from util import clean_license_plate, validate_plate


class ANPRSystem:
    def __init__(self, vehicle_model_path='yolov8n.pt', plate_model_path=None):
        """
        Initialize ANPR system.
        vehicle_model_path: path to YOLOv8 vehicle detection model.
        plate_model_path: path to custom YOLOv8 license plate detection model.
        """
        print("Loading YOLOv8 vehicle detector...")
        self.vehicle_detector = YOLO(vehicle_model_path)

        if plate_model_path and os.path.exists(plate_model_path):
            print("Loading YOLOv8 plate detector...")
            self.plate_detector = YOLO(plate_model_path)
        else:
            print("Warning: No plate detector found. Using fallback vehicle crop for plates.")
            self.plate_detector = None

        self.mot_tracker = Sort(max_age=30, min_hits=3)

        print("Initializing EasyOCR...")
        self.ocr_reader = easyocr.Reader(['en'], gpu=torch.cuda.is_available(), detector=False)

        self.results = []
        self.frame_count = 0

    def process_frame(self, frame):
        """Process a single frame through the entire pipeline."""
        self.frame_count += 1

        vehicle_results = self.vehicle_detector(frame, classes=[2, 3, 5, 7], conf=0.5, verbose=False)
        vehicle_detections = []

        for result in vehicle_results:
            if result.boxes is None:
                continue

            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0].cpu().numpy())
                cls = int(box.cls[0].cpu().numpy())

                if cls in [2, 3, 5, 7]:
                    vehicle_detections.append([x1, y1, x2, y2, conf])

        if not vehicle_detections:
            return frame

        tracked_vehicles = self.mot_tracker.update(np.array(vehicle_detections))

        for track in tracked_vehicles:
            x1, y1, x2, y2, track_id = track
            track_id = int(track_id)
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            vehicle_crop = frame[y1:y2, x1:x2]
            if vehicle_crop.size == 0:
                continue

            plate_crop, plate_bbox, plate_conf = self.detect_plate(frame, vehicle_crop, x1, y1)
            plate_text, license_score = self.read_plate(plate_crop)
            if not plate_text:
                continue

            license_plate = clean_license_plate(plate_text)
            license_plate, validation_score = validate_plate(license_plate)
            license_score = max(license_score, validation_score)

            self.results.append({
                "frame_number": self.frame_count,
                "car_id": track_id,
                "car_bbox": [x1, y1, x2, y2],
                "license_plate_bbox": [float(value) for value in plate_bbox],
                "license_plate_bbox_score": float(plate_conf),
                "license_number": license_plate,
                "license_number_score": float(license_score),
            })

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            bx1, by1, bx2, by2 = [int(value) for value in plate_bbox]
            cv2.rectangle(frame, (bx1, by1), (bx2, by2), (255, 0, 0), 2)
            cv2.putText(
                frame,
                license_plate,
                (x1, max(20, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2,
            )

        return frame

    def detect_plate(self, frame, vehicle_crop, vehicle_x, vehicle_y):
        """Return the best plate crop, global plate bbox, and detector confidence."""
        if self.plate_detector is None:
            return self.fallback_plate_crop(vehicle_crop, vehicle_x, vehicle_y)

        plate_results = self.plate_detector(vehicle_crop, conf=0.05, verbose=False)
        plate_detections = []

        for result in plate_results:
            if result.boxes is None:
                continue

            for box in result.boxes:
                px1, py1, px2, py2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0].cpu().numpy())
                gx1 = vehicle_x + px1
                gy1 = vehicle_y + py1
                gx2 = vehicle_x + px2
                gy2 = vehicle_y + py2
                plate_detections.append([gx1, gy1, gx2, gy2, conf])

        if plate_detections:
            best_plate = max(plate_detections, key=lambda detection: detection[4])
            gx1, gy1, gx2, gy2, plate_conf = best_plate
            plate_crop = frame[int(gy1):int(gy2), int(gx1):int(gx2)]
            return plate_crop, [gx1, gy1, gx2, gy2], plate_conf

        return self.fallback_plate_crop(vehicle_crop, vehicle_x, vehicle_y)

    def fallback_plate_crop(self, vehicle_crop, vehicle_x, vehicle_y):
        height, width = vehicle_crop.shape[:2]
        local_x1 = int(width * 0.2)
        local_y1 = int(height * 0.5)
        local_x2 = int(width * 0.8)
        local_y2 = height
        plate_crop = vehicle_crop[local_y1:local_y2, local_x1:local_x2]
        plate_bbox = [
            vehicle_x + local_x1,
            vehicle_y + local_y1,
            vehicle_x + local_x2,
            vehicle_y + local_y2,
        ]
        return plate_crop, plate_bbox, 0.0

    def read_plate(self, plate_crop):
        if plate_crop.size == 0:
            return "", 0.0

        ocr_results = self.ocr_reader.recognize(
            plate_crop,
            allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
        )
        if not ocr_results:
            return "", 0.0

        plate_text = ocr_results[0][1]
        license_score = float(ocr_results[0][2])
        return plate_text, license_score

    def save_results(self, output_csv='test.csv'):
        """Save results to CSV."""
        df = pd.DataFrame(self.results)
        df.to_csv(output_csv, index=False)
        print(f"Results saved to {output_csv}")
        return df

    def process_video(
        self,
        input_video,
        output_video=None,
        output_csv='test.csv',
        target_size=(1280, 720),
        progress_callback=None,
    ):
        """Process an entire video."""
        self.results = []
        self.frame_count = 0
        self.mot_tracker = Sort(max_age=30, min_hits=3)

        cap = cv2.VideoCapture(input_video)
        if not cap.isOpened():
            raise FileNotFoundError(f"Could not open input video: {input_video}")

        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        output_size = target_size or (width, height)

        out = None
        if output_video:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_video, fourcc, fps, output_size)

        print(f"Processing video: {total_frames} frames at {fps} FPS")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if target_size:
                frame = cv2.resize(frame, (1280, 720))

            processed_frame = self.process_frame(frame)

            if out is not None:
                out.write(processed_frame)

            if self.frame_count % 30 == 0:
                print(f"Processed {self.frame_count}/{total_frames} frames")
                if progress_callback is not None:
                    progress_callback(self.frame_count, total_frames)

        cap.release()
        if out is not None:
            out.release()

        if progress_callback is not None:
            progress_callback(self.frame_count, total_frames)

        return self.save_results(output_csv)


def resolve_input_video(requested_path):
    requested = Path(requested_path)
    if requested.exists():
        return requested

    candidates = [
        requested.with_name(f"{requested.name}.mp4"),
        Path("input_video.mp4.mp4"),
    ]

    for candidate in candidates:
        if candidate.exists():
            print(f"Input video {requested} not found. Using {candidate} instead.")
            return candidate

    return None


def main():
    parser = argparse.ArgumentParser(description="Run ANPR on a video file.")
    parser.add_argument(
        "input_video",
        nargs="?",
        default="input_video.mp4",
        help="Path to the input video. Defaults to input_video.mp4.",
    )
    parser.add_argument(
        "--output",
        default="output_processed.mp4",
        help="Path for the processed output video.",
    )
    args = parser.parse_args()

    input_video = resolve_input_video(args.input_video)
    if input_video is None:
        print(f"Input video {args.input_video} not found!")
        print("Tip: this folder currently contains input_video.mp4.mp4.")
        return

    anpr = ANPRSystem(
        vehicle_model_path='yolov8n.pt',
        plate_model_path='best.pt',
    )

    results = anpr.process_video(str(input_video), args.output)
    print("\nProcessing complete!")
    print(results.head())


if __name__ == "__main__":
    main()
