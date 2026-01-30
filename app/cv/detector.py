import torch
import torchvision
from torchvision.transforms import functional as F


class ObjectDetector:
    def __init__(self, score_threshold: float = 0.5):
        self.device = torch.device("cpu")

        # Load a pretrained detector
        self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
            weights="DEFAULT"
        )
        self.model.to(self.device)
        self.model.eval()

        self.score_threshold = score_threshold

        # COCO class names (partial, enough for traffic)
        self.class_names = {
            1: "person",
            2: "bicycle",
            3: "car",
            4: "motorcycle",
            6: "bus",
            8: "truck",
        }

    def detect(self, frame):
        """
        Run object detection on a single frame (OpenCV BGR image)
        """
        # Convert BGR (OpenCV) -> RGB
        frame_rgb = frame[:, :, ::-1].copy()

        # Convert to tensor
        image_tensor = F.to_tensor(frame_rgb).to(self.device)

        with torch.no_grad():
            outputs = self.model([image_tensor])[0]

        detections = []

        for box, label, score in zip(
            outputs["boxes"], outputs["labels"], outputs["scores"]
        ):
            if score < self.score_threshold:
                continue

            label_id = label.item()
            if label_id not in self.class_names:
                continue

            detections.append({
                "bbox": box.cpu().numpy().astype(int),
                "label": self.class_names[label_id],
                "score": float(score),
            })

        return detections