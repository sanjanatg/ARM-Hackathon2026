import cv2
import numpy as np
import onnxruntime as ort

class AnomalyDetector:
    def __init__(self, model_path, conf_thresh=0.15, iou_thresh=0.45):
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh

        self.session = ort.InferenceSession(
            model_path,
            providers=["CPUExecutionProvider"]
        )

        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

        self.input_size = 640
        self.class_names = ["pothole"]

    def preprocess(self, frame):
        h, w, _ = frame.shape
        img = cv2.resize(frame, (self.input_size, self.input_size))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))  # CHW
        img = np.expand_dims(img, axis=0)
        return img, h, w

    def detect(self, frame):
        img, orig_h, orig_w = self.preprocess(frame)

        output = self.session.run(
            [self.output_name],
            {self.input_name: img}
        )[0]

        # Shape: (1, 5, 8400) → (8400, 5)
        preds = output[0].transpose()

        boxes = []
        scores = []

        for det in preds:
            cx, cy, w, h, score = det   # ✅ ONLY 5 VALUES

            if score < self.conf_thresh:
                continue

            # Scale from 640x640 → original image
            cx = cx * orig_w / self.input_size
            cy = cy * orig_h / self.input_size
            w  = w  * orig_w / self.input_size
            h  = h  * orig_h / self.input_size

            x1 = int(cx - w / 2)
            y1 = int(cy - h / 2)
            x2 = int(cx + w / 2)
            y2 = int(cy + h / 2)

            # Clamp to image bounds
            x1 = max(0, min(orig_w, x1))
            y1 = max(0, min(orig_h, y1))
            x2 = max(0, min(orig_w, x2))
            y2 = max(0, min(orig_h, y2))

            boxes.append([x1, y1, x2 - x1, y2 - y1])
            scores.append(float(score))

        indices = cv2.dnn.NMSBoxes(
            boxes, scores, self.conf_thresh, self.iou_thresh
        )

        results = []
        if len(indices) > 0:
            for i in indices.flatten():
                x, y, w, h = boxes[i]
                results.append({
                    "bbox": (x, y, x + w, y + h),
                    "confidence": scores[i],
                    "label": "pothole"
                })

        return results


