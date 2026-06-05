import os
import csv
import json
import threading
import time
from functools import wraps
from datetime import datetime
from pathlib import Path
from uuid import uuid4

import cv2
import numpy as np
import pandas as pd
from flask import Flask, Response, flash, jsonify, redirect, render_template, request, send_from_directory, session, url_for
from werkzeug.security import check_password_hash, generate_password_hash
from werkzeug.utils import secure_filename

from main import ANPRSystem


BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "web_uploads"
RESULT_DIR = BASE_DIR / "web_results"
CONFIG_DIR = BASE_DIR / "config"
AUTH_FILE = CONFIG_DIR / "auth.json"
AUTH_CSV_FILE = CONFIG_DIR / "users.csv"
ALLOWED_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}

app = Flask(__name__)
app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY", "dev-secret-key")
app.config["MAX_CONTENT_LENGTH"] = 512 * 1024 * 1024

UPLOAD_DIR.mkdir(exist_ok=True)
RESULT_DIR.mkdir(exist_ok=True)
CONFIG_DIR.mkdir(exist_ok=True)

jobs = {}
jobs_lock = threading.Lock()
anpr_lock = threading.Lock()
anpr_system = None


def login_required(view):
    @wraps(view)
    def wrapped_view(*args, **kwargs):
        if not session.get("authenticated"):
            return redirect(url_for("login", next=request.path))
        return view(*args, **kwargs)

    return wrapped_view


def auth_configured():
    return bool(load_auth()["users"])


def load_auth():
    if not AUTH_FILE.exists():
        return {"users": []}

    auth_data = json.loads(AUTH_FILE.read_text(encoding="utf-8"))
    if "users" in auth_data:
        return auth_data

    migrated_user = {
        "id": uuid4().hex,
        "username": auth_data.get("username", ""),
        "email": auth_data.get("email", ""),
        "password_hash": auth_data.get("password_hash", ""),
        "created_at": auth_data.get("updated_at", datetime.now().isoformat(timespec="seconds")),
        "updated_at": auth_data.get("updated_at", datetime.now().isoformat(timespec="seconds")),
    }
    migrated_data = {"users": [migrated_user] if migrated_user["username"] else []}
    save_auth_data(migrated_data)
    return migrated_data


def save_auth_data(auth_data):
    AUTH_FILE.write_text(json.dumps(auth_data, indent=2), encoding="utf-8")
    save_users_csv(auth_data["users"])


def save_users_csv(users):
    fieldnames = ["id", "username", "email", "password_hash", "created_at", "updated_at"]
    with AUTH_CSV_FILE.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for user in users:
            writer.writerow({field: user.get(field, "") for field in fieldnames})


def find_user_by_identifier(identifier):
    identifier = identifier.strip()
    identifier_lower = identifier.lower()
    for user in load_auth()["users"]:
        if user.get("id") == identifier:
            return user
        if user.get("username") == identifier:
            return user
        if user.get("email", "").lower() == identifier_lower:
            return user
    return None


def username_or_email_exists(username, email, ignore_user_id=None):
    username_lower = username.lower()
    email_lower = email.lower()
    for user in load_auth()["users"]:
        if ignore_user_id and user.get("id") == ignore_user_id:
            continue
        if user.get("username", "").lower() == username_lower:
            return "Username already exists."
        if email and user.get("email", "").lower() == email_lower:
            return "Email already exists."
    return ""


def create_user(username, email, password):
    auth_data = load_auth()
    auth_record = {
        "id": uuid4().hex,
        "username": username,
        "email": email,
        "password_hash": generate_password_hash(password),
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "updated_at": datetime.now().isoformat(timespec="seconds"),
    }
    auth_data["users"].append(auth_record)
    save_auth_data(auth_data)
    return auth_record


def update_user(user_id, username, email, password):
    auth_data = load_auth()
    for user in auth_data["users"]:
        if user.get("id") == user_id:
            user["username"] = username
            user["email"] = email
            user["password_hash"] = generate_password_hash(password)
            user["updated_at"] = datetime.now().isoformat(timespec="seconds")
            save_auth_data(auth_data)
            return user
    return None


def delete_user(user_id):
    auth_data = load_auth()
    remaining_users = [user for user in auth_data["users"] if user.get("id") != user_id]
    if len(remaining_users) == len(auth_data["users"]):
        return False

    auth_data["users"] = remaining_users
    save_auth_data(auth_data)
    return True


def verify_credentials(identifier, password):
    auth_record = find_user_by_identifier(identifier)
    if auth_record and check_password_hash(auth_record["password_hash"], password):
        return auth_record
    return None


def valid_email(email):
    return "@" in email and "." in email.rsplit("@", 1)[-1]


def allowed_video(filename):
    return Path(filename).suffix.lower() in ALLOWED_EXTENSIONS


def get_anpr_system():
    global anpr_system
    if anpr_system is None:
        anpr_system = ANPRSystem(
            vehicle_model_path=str(BASE_DIR / "yolov8n.pt"),
            plate_model_path=str(BASE_DIR / "best.pt"),
        )
    return anpr_system


def persist_job(job):
    job_dir = RESULT_DIR / job["id"]
    job_dir.mkdir(exist_ok=True)
    metadata = {
        key: value
        for key, value in job.items()
        if key not in {"input_path", "output_path", "csv_path"}
    }
    (job_dir / "job.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def load_existing_jobs():
    with jobs_lock:
        for job_dir in RESULT_DIR.iterdir():
            if not job_dir.is_dir():
                continue

            job_id = job_dir.name
            metadata_path = job_dir / "job.json"
            output_path = job_dir / "processed.mp4"
            csv_path = job_dir / "results.csv"
            if not output_path.exists() or not csv_path.exists():
                continue

            if metadata_path.exists():
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            else:
                metadata = {
                    "id": job_id,
                    "filename": "processed.mp4",
                    "status": "complete",
                    "progress": 100,
                    "current_frame": 0,
                    "total_frames": 0,
                    "error": "",
                    "created_at": datetime.fromtimestamp(output_path.stat().st_mtime).isoformat(timespec="seconds"),
                    "started_at": "",
                    "finished_at": datetime.fromtimestamp(output_path.stat().st_mtime).isoformat(timespec="seconds"),
                }

            metadata.update({
                "id": job_id,
                "input_path": "",
                "output_path": str(output_path),
                "csv_path": str(csv_path),
            })
            jobs[job_id] = metadata


def update_job(job_id, **values):
    with jobs_lock:
        jobs[job_id].update(values)
        job = dict(jobs[job_id])
    persist_job(job)


def job_snapshot(job_id):
    with jobs_lock:
        job = jobs.get(job_id)
        return dict(job) if job else None


def result_summary(csv_path):
    if not csv_path.exists():
        return [], [], 0

    results = pd.read_csv(csv_path)
    if results.empty:
        return [], [], 0

    rows = results.sort_values("frame_number").head(100).to_dict("records")
    unique_plates = []
    if "license_number" in results:
        unique_plates = sorted(results["license_number"].dropna().unique().tolist())
    return rows, unique_plates, len(results)


def processed_video_path(job_id):
    job = job_snapshot(job_id)
    if job is None:
        return None

    output_path = Path(job["output_path"])
    if output_path.exists():
        return output_path

    fallback_path = RESULT_DIR / job_id / "processed.mp4"
    return fallback_path if fallback_path.exists() else None


def make_message_frame(message):
    frame = 255 * np.ones((420, 760, 3), dtype="uint8")
    cv2.putText(frame, message, (34, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (20, 33, 61), 2)
    ok, buffer = cv2.imencode(".jpg", frame)
    return buffer.tobytes() if ok else b""


def process_job(job_id):
    job = job_snapshot(job_id)
    if not job:
        return

    def progress(current_frame, total_frames):
        percent = 0
        if total_frames:
            percent = min(99, int((current_frame / total_frames) * 100))
        update_job(
            job_id,
            progress=percent,
            current_frame=current_frame,
            total_frames=total_frames,
        )

    update_job(job_id, status="processing", progress=1, started_at=datetime.now().isoformat(timespec="seconds"))

    try:
        with anpr_lock:
            get_anpr_system().process_video(
                job["input_path"],
                job["output_path"],
                output_csv=job["csv_path"],
                progress_callback=progress,
            )
        update_job(job_id, status="complete", progress=100, finished_at=datetime.now().isoformat(timespec="seconds"))
    except Exception as exc:
        update_job(job_id, status="failed", error=str(exc), finished_at=datetime.now().isoformat(timespec="seconds"))


@app.route("/login", methods=["GET", "POST"])
def login():
    if not auth_configured():
        return redirect(url_for("signup"))

    if session.get("authenticated"):
        return redirect(url_for("index"))

    if request.method == "GET":
        return render_template("login.html")

    username = request.form.get("username", "").strip()
    password = request.form.get("password", "")
    auth_record = verify_credentials(username, password)
    if auth_record:
        session["authenticated"] = True
        session["user_id"] = auth_record.get("id", "")
        session["username"] = auth_record.get("username", username)
        session["email"] = auth_record.get("email", "")
        return redirect(request.args.get("next") or url_for("index"))

    flash("Invalid username or password.")
    return render_template("login.html"), 401


@app.route("/setup", methods=["GET", "POST"])
@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "GET":
        return render_template("setup.html")

    username = request.form.get("username", "").strip()
    email = request.form.get("email", "").strip()
    password = request.form.get("password", "")
    confirm_password = request.form.get("confirm_password", "")

    if not username:
        flash("Enter a username.")
        return render_template("setup.html"), 400
    if not email or not valid_email(email):
        flash("Enter a valid email address.")
        return render_template("setup.html"), 400
    if not password:
        flash("Enter a password.")
        return render_template("setup.html"), 400
    if password != confirm_password:
        flash("Passwords do not match.")
        return render_template("setup.html"), 400
    duplicate_message = username_or_email_exists(username, email)
    if duplicate_message:
        flash(duplicate_message)
        return render_template("setup.html"), 400

    auth_record = create_user(username, email, password)
    session["authenticated"] = True
    session["user_id"] = auth_record.get("id", "")
    session["username"] = username
    session["email"] = email
    flash("Account created successfully.")
    return redirect(url_for("index"))


@app.route("/logout", methods=["POST"])
def logout():
    session.clear()
    return redirect(url_for("login" if auth_configured() else "signup"))


@app.route("/account", methods=["GET", "POST"])
@login_required
def account_settings():
    auth_record = find_user_by_identifier(session.get("user_id", "")) if session.get("user_id") else None
    if auth_record is None:
        auth_record = find_user_by_identifier(session.get("username", ""))
    if auth_record is None:
        session.clear()
        return redirect(url_for("signup"))

    if request.method == "GET":
        return render_template(
            "account.html",
            current_username=auth_record.get("username", ""),
            current_email=auth_record.get("email", ""),
        )

    username = request.form.get("username", "").strip()
    email = request.form.get("email", "").strip()
    current_password = request.form.get("current_password", "")
    new_password = request.form.get("new_password", "")
    confirm_password = request.form.get("confirm_password", "")

    if not username:
        flash("Enter a username.")
        return render_template("account.html", current_username=auth_record.get("username", ""), current_email=auth_record.get("email", "")), 400
    if not email or not valid_email(email):
        flash("Enter a valid email address.")
        return render_template("account.html", current_username=auth_record.get("username", ""), current_email=auth_record.get("email", "")), 400
    if not check_password_hash(auth_record["password_hash"], current_password):
        flash("Current password is incorrect.")
        return render_template("account.html", current_username=auth_record.get("username", ""), current_email=auth_record.get("email", "")), 401
    if new_password != confirm_password:
        flash("New passwords do not match.")
        return render_template("account.html", current_username=auth_record.get("username", ""), current_email=auth_record.get("email", "")), 400
    duplicate_message = username_or_email_exists(username, email, ignore_user_id=auth_record.get("id"))
    if duplicate_message:
        flash(duplicate_message)
        return render_template("account.html", current_username=auth_record.get("username", ""), current_email=auth_record.get("email", "")), 400

    password_to_save = new_password if new_password else current_password
    updated_user = update_user(auth_record.get("id"), username, email, password_to_save)
    if updated_user is None:
        session.clear()
        return redirect(url_for("signup"))
    session["username"] = username
    session["email"] = email
    flash("Account settings updated.")
    return redirect(url_for("account_settings"))


@app.route("/account/delete", methods=["POST"])
@login_required
def delete_account():
    auth_record = find_user_by_identifier(session.get("user_id", "")) if session.get("user_id") else None
    if auth_record is None:
        auth_record = find_user_by_identifier(session.get("username", ""))
    if auth_record is None:
        session.clear()
        return redirect(url_for("signup"))

    delete_password = request.form.get("delete_password", "")
    if not check_password_hash(auth_record["password_hash"], delete_password):
        flash("Password is incorrect. Account was not deleted.")
        return redirect(url_for("account_settings"))

    delete_user(auth_record.get("id"))
    session.clear()
    flash("Account deleted.")
    return redirect(url_for("login" if auth_configured() else "signup"))


@app.route("/", methods=["GET"])
@login_required
def index():
    with jobs_lock:
        recent_jobs = sorted(jobs.values(), key=lambda item: item["created_at"], reverse=True)[:8]
    return render_template("index.html", jobs=recent_jobs)


@app.route("/live")
@login_required
def live_camera():
    return render_template("live.html")


@app.route("/live-feed")
@login_required
def live_feed():
    camera_index = int(request.args.get("camera", "0"))
    process_every = max(1, int(request.args.get("every", "5")))

    def generate_live_frames():
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            frame = make_message_frame("Camera not found. Check webcam permission or camera index.")
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
            return

        frame_number = 0
        last_processed_frame = None

        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                frame = cv2.resize(frame, (960, 540))
                frame_number += 1

                if frame_number % process_every == 0 or last_processed_frame is None:
                    with anpr_lock:
                        system = get_anpr_system()
                        processed_frame = system.process_frame(frame.copy())
                        system.results = []
                    last_processed_frame = processed_frame
                else:
                    processed_frame = last_processed_frame

                cv2.putText(
                    processed_frame,
                    "LIVE CAMERA",
                    (18, 38),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 255, 255),
                    2,
                )

                ok, buffer = cv2.imencode(".jpg", processed_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 82])
                if not ok:
                    continue

                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n"
        finally:
            cap.release()

    return Response(generate_live_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/upload", methods=["POST"])
@login_required
def upload():
    uploaded_file = request.files.get("video")
    if uploaded_file is None or uploaded_file.filename == "":
        flash("Choose a video file first.")
        return redirect(url_for("index"))

    if not allowed_video(uploaded_file.filename):
        flash("Upload an MP4, AVI, MOV, or MKV video.")
        return redirect(url_for("index"))

    job_id = uuid4().hex
    job_dir = RESULT_DIR / job_id
    job_dir.mkdir()

    safe_name = secure_filename(uploaded_file.filename)
    input_path = UPLOAD_DIR / f"{job_id}_{safe_name}"
    output_path = job_dir / "processed.mp4"
    csv_path = job_dir / "results.csv"
    uploaded_file.save(input_path)

    with jobs_lock:
        jobs[job_id] = {
            "id": job_id,
            "filename": safe_name,
            "input_path": str(input_path),
            "output_path": str(output_path),
            "csv_path": str(csv_path),
            "status": "queued",
            "progress": 0,
            "current_frame": 0,
            "total_frames": 0,
            "error": "",
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "started_at": "",
            "finished_at": "",
        }
        new_job = dict(jobs[job_id])
    persist_job(new_job)

    thread = threading.Thread(target=process_job, args=(job_id,), daemon=True)
    thread.start()
    return redirect(url_for("job_status", job_id=job_id))


@app.route("/jobs/<job_id>")
@login_required
def job_status(job_id):
    job = job_snapshot(job_id)
    if job is None:
        flash("Job not found.")
        return redirect(url_for("index"))
    return render_template("job.html", job=job)


@app.route("/api/jobs/<job_id>")
@login_required
def job_api(job_id):
    job = job_snapshot(job_id)
    if job is None:
        return jsonify({"error": "Job not found"}), 404
    return jsonify({
        "id": job["id"],
        "filename": job["filename"],
        "status": job["status"],
        "progress": job["progress"],
        "current_frame": job["current_frame"],
        "total_frames": job["total_frames"],
        "error": job["error"],
        "result_url": url_for("result", job_id=job_id) if job["status"] == "complete" else "",
    })


@app.route("/results/<job_id>")
@login_required
def result(job_id):
    job = job_snapshot(job_id)
    if job is None:
        flash("Job not found.")
        return redirect(url_for("index"))

    if job["status"] != "complete":
        return redirect(url_for("job_status", job_id=job_id))

    rows, unique_plates, total_rows = result_summary(Path(job["csv_path"]))
    return render_template(
        "result.html",
        job=job,
        rows=rows,
        unique_plates=unique_plates,
        total_rows=total_rows,
    )


@app.route("/plates/<job_id>")
@login_required
def plate_candidates(job_id):
    job = job_snapshot(job_id)
    if job is None:
        flash("Job not found.")
        return redirect(url_for("index"))

    if job["status"] != "complete":
        return redirect(url_for("job_status", job_id=job_id))

    rows, unique_plates, total_rows = result_summary(Path(job["csv_path"]))
    return render_template(
        "plates.html",
        job=job,
        rows=rows,
        unique_plates=unique_plates,
        total_rows=total_rows,
    )


@app.route("/preview-results/<job_id>")
@login_required
def results_preview(job_id):
    job = job_snapshot(job_id)
    if job is None:
        flash("Job not found.")
        return redirect(url_for("index"))

    if job["status"] != "complete":
        return redirect(url_for("job_status", job_id=job_id))

    rows, unique_plates, total_rows = result_summary(Path(job["csv_path"]))
    return render_template(
        "results_preview.html",
        job=job,
        rows=rows,
        unique_plates=unique_plates,
        total_rows=total_rows,
    )


@app.route("/files/<job_id>/<path:filename>")
@login_required
def result_file(job_id, filename):
    mimetype = "video/mp4" if filename.lower().endswith(".mp4") else None
    return send_from_directory(RESULT_DIR / job_id, filename, as_attachment=False, mimetype=mimetype)


@app.route("/preview/<job_id>")
@login_required
def video_preview(job_id):
    video_path = processed_video_path(job_id)
    if video_path is None:
        return "Processed video not found.", 404

    def generate_frames():
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS) or 15
        delay = min(1 / fps, 0.08)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            ok, buffer = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 82])
            if not ok:
                continue

            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n"
            )
            time.sleep(delay)

        cap.release()

    return Response(generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/download/<job_id>/<path:filename>")
@login_required
def download_file(job_id, filename):
    return send_from_directory(RESULT_DIR / job_id, filename, as_attachment=True)


if __name__ == "__main__":
    save_auth_data(load_auth())
    load_existing_jobs()
    port = int(os.environ.get("PORT", "5000"))
    debug_enabled = os.environ.get("FLASK_DEBUG", "0") == "1"
    app.run(host="0.0.0.0", port=port, debug=debug_enabled, threaded=True, use_reloader=False)
