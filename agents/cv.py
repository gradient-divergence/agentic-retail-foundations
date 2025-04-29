"""
Module: agents.cv

Contains the ShelfMonitoringAgent class for computer vision-based shelf monitoring in retail.
"""

from typing import Any
import asyncio
import time
from datetime import datetime
import cv2
import numpy as np
import tensorflow as tf
import logging


class ShelfMonitoringAgent:
    """
    Agent for monitoring retail shelves using computer vision.
    Processes camera feeds to detect products, compares with planograms, and reports issues.
    """

    def __init__(
        self,
        model_path: str,
        planogram_database,
        inventory_system,
        camera_stream_urls: dict[str, str],
        confidence_threshold: float = 0.65,
        check_frequency_seconds: int = 300,
    ):
        """Initialize the shelf monitoring agent."""
        # Try to load the TensorFlow SavedModel.  In demo / documentation settings a real
        # model may not be available, so fall back to a *no-op* model that returns empty
        # detections.  This lets the rest of the agent run without crashing while still
        # warning the user that no detections will be produced.
        try:
            self.detection_model = tf.saved_model.load(model_path)
        except (OSError, ValueError) as e:
            logging.warning(
                "ShelfMonitoringAgent: Could not load model from '%s': %s. "
                "Falling back to a dummy detection model (no detections will be produced).",
                model_path,
                e,
            )

            class _DummyModel:
                """A minimal stand-in that mimics the TF object-detection API."""

                def __call__(self, inputs, *args, **kwargs):  # noqa: D401 – simple stub
                    import tensorflow as tf

                    batch = inputs.shape[0] if hasattr(inputs, "shape") else 1
                    empty = tf.zeros([batch, 1, 4], dtype=tf.float32)
                    zeros = tf.zeros([batch, 1], dtype=tf.float32)
                    return {
                        "detection_boxes": empty,
                        "detection_classes": zeros,
                        "detection_scores": zeros,
                    }

            self.detection_model = _DummyModel()
        self.planogram_db = planogram_database
        self.inventory_system = inventory_system
        self.camera_streams = camera_stream_urls
        self.active_streams = {}
        self.confidence_threshold = confidence_threshold
        self.check_frequency = check_frequency_seconds
        self.last_check_times = {}
        self.detected_issues = {}

    async def start_monitoring_section(self, location_id: str, section_id: str):
        """Begin monitoring a specific shelf section at a location."""
        camera_id = await self.planogram_db.get_section_camera(location_id, section_id)
        if not camera_id or camera_id not in self.camera_streams:
            print(
                f"No camera configured for section {section_id} at location {location_id}"
            )
            return
        if camera_id not in self.active_streams:
            self.active_streams[camera_id] = cv2.VideoCapture(
                self.camera_streams[camera_id]
            )
        self.last_check_times[section_id] = 0
        self.detected_issues[section_id] = []
        await self._monitor_section_loop(location_id, section_id)

    async def stop_monitoring_section(self, location_id: str, section_id: str):
        """Stop monitoring a specific shelf section."""
        camera_id = await self.planogram_db.get_section_camera(location_id, section_id)
        if camera_id in self.active_streams:
            self.active_streams[camera_id].release()
            del self.active_streams[camera_id]
        if section_id in self.last_check_times:
            del self.last_check_times[section_id]
        if section_id in self.detected_issues:
            del self.detected_issues[section_id]

    async def _monitor_section_loop(self, location_id: str, section_id: str):
        """Monitoring loop for a shelf section."""
        camera_id = await self.planogram_db.get_section_camera(location_id, section_id)
        stream = self.active_streams.get(camera_id)
        while stream and stream.isOpened():
            current_time = time.time()
            if (
                current_time - self.last_check_times.get(section_id, 0)
                >= self.check_frequency
            ):
                await self._check_section(location_id, section_id, camera_id, stream)
                self.last_check_times[section_id] = current_time
            await asyncio.sleep(1)

    async def _check_section(
        self,
        location_id: str,
        section_id: str,
        camera_id: str,
        stream: cv2.VideoCapture,
    ):
        """Analyze current shelf state for a specific section."""
        planogram = await self.planogram_db.get_section_planogram(
            location_id, section_id
        )
        if not planogram:
            return
        ret, frame = stream.read()
        if not ret:
            print(f"Failed to read frame from camera {camera_id}")
            return
        input_tensor = self._preprocess_image(frame)
        detections = self.detection_model(input_tensor)
        detected_products = self._process_detections(
            detections, frame.shape[1], frame.shape[0]
        )
        issues = self._compare_with_planogram(detected_products, planogram)
        if issues:
            timestamp = datetime.now().isoformat()
            self.detected_issues[section_id] = issues
            await self._report_issues(location_id, section_id, issues, timestamp)

    def _preprocess_image(self, image: np.ndarray) -> tf.Tensor:
        """Convert image to the format required by the model."""
        input_size = (640, 640)
        image_resized = cv2.resize(image, input_size)
        image_rgb = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)
        image_normalized = image_rgb / 255.0
        input_tensor = tf.expand_dims(image_normalized, 0)
        return input_tensor

    def _process_detections(
        self, detections: dict, img_w: int, img_h: int
    ) -> list[dict[str, Any]]:
        """Process raw detections into structured product data."""
        detection_boxes = detections["detection_boxes"][0].numpy()
        detection_classes = detections["detection_classes"][0].numpy().astype(np.int32)
        detection_scores = detections["detection_scores"][0].numpy()
        class_mapping = self._get_class_mapping()
        products = []
        for i in range(len(detection_scores)):
            if detection_scores[i] >= self.confidence_threshold:
                box = detection_boxes[i]
                ymin, xmin, ymax, xmax = box
                box_pixel = [
                    int(ymin * img_h),
                    int(xmin * img_w),
                    int(ymax * img_h),
                    int(xmax * img_w),
                ]
                class_id = detection_classes[i]
                if class_id in class_mapping:
                    product_id = class_mapping[class_id]
                    products.append(
                        {
                            "product_id": product_id,
                            "confidence": float(detection_scores[i]),
                            "bounding_box": box_pixel,
                            "shelf_position": {
                                "x": (xmin + xmax) / 2,
                                "y": (ymin + ymax) / 2,
                            },
                        }
                    )
        return products

    def _get_class_mapping(self) -> dict[int, str]:
        """Map model class IDs to product IDs."""
        return {
            1: "SKU123456",
            2: "SKU789012",
        }

    def _compare_with_planogram(
        self,
        detected_products: list[dict[str, Any]],
        planogram: dict[str, Any],
        tol: float = 0.15,
    ) -> list[dict[str, Any]]:
        """Compare detected products with expected planogram."""
        issues = []
        product_counts = {}
        product_positions = {}
        for product in detected_products:
            product_id = product["product_id"]
            if product_id in product_counts:
                product_counts[product_id] += 1
                product_positions[product_id].append(product["shelf_position"])
            else:
                product_counts[product_id] = 1
                product_positions[product_id] = [product["shelf_position"]]
        for expected_product in planogram["products"]:
            product_id = expected_product["product_id"]
            expected_count = expected_product["expected_count"]
            actual_count = product_counts.get(product_id, 0)
            if actual_count < expected_count:
                gap_percentage = (expected_count - actual_count) / expected_count
                issues.append(
                    {
                        "type": "OUT_OF_STOCK" if actual_count == 0 else "LOW_STOCK",
                        "product_id": product_id,
                        "expected_count": expected_count,
                        "actual_count": actual_count,
                        "gap_percentage": gap_percentage,
                        "position": expected_product["position"],
                    }
                )
            if product_id in product_counts:
                del product_counts[product_id]
        for product_id, count in product_counts.items():
            issues.append(
                {
                    "type": "UNEXPECTED_PRODUCT",
                    "product_id": product_id,
                    "count": count,
                    "positions": product_positions[product_id],
                }
            )
        for product in detected_products:
            product_id = product["product_id"]
            for expected_product in planogram["products"]:
                if expected_product["product_id"] == product_id:
                    expected_pos = expected_product["position"]
                    actual_pos = product["shelf_position"]
                    distance = np.sqrt(
                        (expected_pos["x"] - actual_pos["x"]) ** 2
                        + (expected_pos["y"] - actual_pos["y"]) ** 2
                    )
                    if distance > tol:
                        issues.append(
                            {
                                "type": "MISPLACED_PRODUCT",
                                "product_id": product_id,
                                "expected_position": expected_pos,
                                "actual_position": actual_pos,
                                "distance": distance,
                            }
                        )
                    break
        return issues

    async def _report_issues(
        self,
        location_id: str,
        section_id: str,
        issues: list[dict[str, Any]],
        timestamp: str,
    ):
        """Report detected issues to inventory system."""
        issue_summary = {
            "location_id": location_id,
            "section_id": section_id,
            "timestamp": timestamp,
            "issues": issues,
        }
        await self.inventory_system.report_visual_audit(issue_summary)
        print(
            f"[{timestamp}] Detected {len(issues)} issues in section {section_id} at {location_id}"
        )
        for issue in issues:
            print(f"  - {issue['type']}: {issue['product_id']}")

    async def stop_all_monitoring(self):
        """Stop all monitoring and release resources."""
        for stream in self.active_streams.values():
            stream.release()
        self.active_streams.clear()
        self.last_check_times.clear()
        self.detected_issues.clear()
