"""
Module: agents.cv

Contains the ShelfMonitoringAgent class for computer vision-based shelf monitoring in retail.
"""

import asyncio
import logging
import time
from datetime import datetime
from typing import Any, cast

import cv2
import numpy as np
import torch

logger = logging.getLogger(__name__)


class ShelfMonitoringAgent:
    """
    Agent for monitoring retail shelves using computer vision.
    Processes camera feeds to detect products, compares with planograms, and reports issues.
    """

    def __init__(
        self,
        model_path: str,
        planogram_database: Any,
        inventory_system: Any,
        camera_stream_urls: dict[str, str],
        confidence_threshold: float = 0.65,
        check_frequency_seconds: int = 300,
    ) -> None:
        """Initialize the shelf monitoring agent."""
        # A missing detector must not turn an unavailable audit into out-of-stock alerts.
        self.detection_model = torch.jit.load(model_path)
        self.detection_model.eval()
        self.planogram_db = planogram_database
        self.inventory_system = inventory_system
        self.camera_streams = camera_stream_urls
        self.active_streams: dict[str, cv2.VideoCapture] = {}
        self.confidence_threshold = confidence_threshold
        self.check_frequency = check_frequency_seconds
        self.last_check_times: dict[str, float] = {}
        self.detected_issues: dict[str, list[dict[str, Any]]] = {}

    async def start_monitoring_section(self, location_id: str, section_id: str) -> None:
        """Begin monitoring a specific shelf section at a location."""
        camera_id = await self.planogram_db.get_section_camera(location_id, section_id)
        if not camera_id or camera_id not in self.camera_streams:
            logger.warning(
                "No camera configured for section %s at location %s",
                section_id,
                location_id,
            )
            return
        if camera_id not in self.active_streams:
            self.active_streams[camera_id] = cv2.VideoCapture(self.camera_streams[camera_id])
        self.last_check_times[section_id] = 0
        self.detected_issues[section_id] = []
        await self._monitor_section_loop(location_id, section_id)

    async def stop_monitoring_section(self, location_id: str, section_id: str) -> None:
        """Stop monitoring a specific shelf section."""
        camera_id = await self.planogram_db.get_section_camera(location_id, section_id)
        if camera_id in self.active_streams:
            self.active_streams[camera_id].release()
            del self.active_streams[camera_id]
        if section_id in self.last_check_times:
            del self.last_check_times[section_id]
        if section_id in self.detected_issues:
            del self.detected_issues[section_id]

    async def _monitor_section_loop(self, location_id: str, section_id: str) -> None:
        """Monitoring loop for a shelf section."""
        camera_id = await self.planogram_db.get_section_camera(location_id, section_id)
        stream = self.active_streams.get(camera_id)
        while stream and stream.isOpened():
            current_time = time.time()
            if current_time - self.last_check_times.get(section_id, 0) >= self.check_frequency:
                await self._check_section(location_id, section_id, camera_id, stream)
                self.last_check_times[section_id] = current_time
            await asyncio.sleep(1)

    async def _check_section(
        self,
        location_id: str,
        section_id: str,
        camera_id: str,
        stream: cv2.VideoCapture,
    ) -> None:
        """Analyze current shelf state for a specific section."""
        planogram = await self.planogram_db.get_section_planogram(location_id, section_id)
        if not planogram:
            return
        ret, frame = stream.read()
        if not ret:
            logger.warning("Failed to read frame from camera %s", camera_id)
            return
        input_tensor = self._preprocess_image(frame)
        detections = self.detection_model(input_tensor)
        detected_products = self._process_detections(detections, frame.shape[1], frame.shape[0])
        issues = self._compare_with_planogram(detected_products, planogram)
        self.detected_issues[section_id] = issues
        if issues:
            timestamp = datetime.now().isoformat()
            await self._report_issues(location_id, section_id, issues, timestamp)

    def _preprocess_image(self, image: np.ndarray) -> torch.Tensor:
        """Convert image to the format required by the model."""
        input_size = (640, 640)
        image_resized = cv2.resize(image, input_size)
        image_rgb = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)
        image_normalized = image_rgb / 255.0
        input_tensor = torch.from_numpy(image_normalized).float().unsqueeze(0)
        return input_tensor

    def _process_detections(self, detections: dict[str, Any], img_w: int, img_h: int) -> list[dict[str, Any]]:
        """Process raw detections into structured product data."""
        # Assuming model output dictionary values are lists of tensors/objects
        # Access the first element (index 0) which is the tensor for the first (only) batch image
        detection_boxes_tensor = detections["detection_boxes"][0]
        detection_classes_tensor = detections["detection_classes"][0]
        detection_scores_tensor = detections["detection_scores"][0]

        # Convert tensors to numpy arrays
        detection_boxes_np = self._to_numpy(detection_boxes_tensor)
        detection_classes_np = self._to_numpy(detection_classes_tensor).astype(np.int32)
        detection_scores_np = self._to_numpy(detection_scores_tensor)

        class_mapping = self._get_class_mapping()
        products = []
        # Loop through detections FOR THE FIRST IMAGE in the batch (index 0)
        num_detections = detection_scores_np.shape[0]  # Number of detections for this image
        for i in range(num_detections):
            if detection_scores_np[i] >= self.confidence_threshold:
                box = detection_boxes_np[i]
                ymin, xmin, ymax, xmax = box
                box_pixel = [
                    int(ymin * img_h),
                    int(xmin * img_w),
                    int(ymax * img_h),
                    int(xmax * img_w),
                ]
                class_id = detection_classes_np[i]
                if class_id in class_mapping:
                    product_id = class_mapping[class_id]
                    products.append(
                        {
                            "product_id": product_id,
                            "confidence": float(detection_scores_np[i]),
                            "bounding_box": box_pixel,
                            "shelf_position": {
                                "x": (xmin + xmax) / 2,
                                "y": (ymin + ymax) / 2,
                            },
                        }
                    )
        return products

    @staticmethod
    def _to_numpy(value: Any) -> np.ndarray:
        """Convert torch tensors (or numpy-like values) into numpy arrays."""
        if torch.is_tensor(value):
            return cast(np.ndarray, value.detach().cpu().numpy())
        if hasattr(value, "numpy"):
            return cast(np.ndarray, value.numpy())
        return cast(np.ndarray, np.asarray(value))

    def _get_class_mapping(self) -> dict[int, str]:
        """Map model class IDs to product IDs."""
        return {
            1: "SKU123456",
            2: "SKU789012",
        }

    def _compare_with_planogram(  # noqa: C901
        self,
        detected_products: list[dict[str, Any]],
        planogram: dict[str, Any],
        tol: float = 0.15,
    ) -> list[dict[str, Any]]:
        """Compare detected products with expected planogram."""
        issues = []
        product_counts: dict[str, int] = {}
        product_positions: dict[str, list[dict[str, float]]] = {}
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
    ) -> None:
        """Report detected issues to inventory system."""
        issue_summary = {
            "location_id": location_id,
            "section_id": section_id,
            "timestamp": timestamp,
            "issues": issues,
        }
        await self.inventory_system.report_visual_audit(issue_summary)
        logger.info(
            "[%s] Detected %s issues in section %s at %s",
            timestamp,
            len(issues),
            section_id,
            location_id,
        )
        for issue in issues:
            logger.info("Issue: %s (%s)", issue["type"], issue["product_id"])

    async def stop_all_monitoring(self) -> None:
        """Stop all monitoring and release resources."""
        for stream in self.active_streams.values():
            stream.release()
        self.active_streams.clear()
        self.last_check_times.clear()
        self.detected_issues.clear()
