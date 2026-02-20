from flask import Flask, render_template, Response
import cv2
import time
from detector import AnomalyDetector

app = Flask(__name__)

detector = AnomalyDetector(
    "best.onnx",
    conf_thresh=0.15
)

cap = cv2.VideoCapture("C:/Users/LENOVO/Documents/BharatAi/road.mp4")  # Change to your IP camera URL


def generate_frames():
    prev = time.time()

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        detections = detector.detect(frame)

        for d in detections:
            x1, y1, x2, y2 = d["bbox"]
            conf = d["confidence"]

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(
                frame,
                f"pothole {conf:.2f}",
                (x1, y1 - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2
            )

        fps = int(1 / (time.time() - prev))
        prev = time.time()

        cv2.putText(
            frame,
            f"FPS: {fps}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

        ret, buf = cv2.imencode(".jpg", frame)
        frame = buf.tobytes()

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
        )


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/video_feed")
def video_feed():
    return Response(
        generate_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)