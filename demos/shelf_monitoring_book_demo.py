# region book:shelf-monitoring-imports
"""ShelfMonitoringAgent for real-time retail shelf analysis using computer vision.

This module provides functionality for monitoring retail shelves using
camera streams and computer vision models to detect product placement,
stock levels, and planogram compliance issues.
"""

# Standard library imports
import asyncio
import time
from datetime import datetime

# Third-party imports
import cv2
import numpy as np
import torch
from pydantic import BaseModel, JsonValue


class ShelfPosition(BaseModel):
    x: float
    y: float


class DetectedProduct(BaseModel):
    product_id: str
    confidence: float
    bounding_box: list[int]
    shelf_position: ShelfPosition


class PlanogramProduct(BaseModel):
    product_id: str
    expected_count: int
    position: ShelfPosition


class Planogram(BaseModel):
    products: list[PlanogramProduct]


class ShelfIssue(BaseModel):
    type: str
    product_id: str
    details: dict[str, JsonValue]


class IssueSummary(BaseModel):
    location_id: str
    section_id: str
    timestamp: str
    issues: list[ShelfIssue]


# endregion book:shelf-monitoring-imports


# region book:shelf-monitoring-class
class ShelfMonitoringAgent:
    """Agent for monitoring retail shelves using computer vision.

    This class processes camera feeds to detect products on shelves,
    compares with expected planograms, and reports issues such as
    out-of-stock conditions or misplaced products.
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
        """Initialize the shelf monitoring agent.

        Args:
            model_path: Path to the saved object detection model
            planogram_database: Database connector for planogram info
            inventory_system: System connector for inventory updates
            camera_stream_urls: Dict mapping camera IDs to stream URLs
            confidence_threshold: Min confidence for detection (0-1)
            check_frequency_seconds: How often to check each section
        """
        # Load the object detection model (TorchScript)
        self.detection_model = torch.jit.load(model_path)
        self.detection_model.eval()
        # Connect to retail systems
        self.planogram_db = planogram_database
        self.inventory_system = inventory_system
        # Store camera stream information
        self.camera_streams = camera_stream_urls
        self.active_streams = {}
        # Configuration
        self.confidence_threshold = confidence_threshold
        self.check_frequency = check_frequency_seconds
        # Monitoring state
        self.last_check_times = {}
        self.detected_issues = {}

    # endregion book:shelf-monitoring-class

    # region book:shelf-monitoring-start-monitoring
    async def start_monitoring(self, location_id: str, section_ids: list[str]):
        """Begin monitoring specified shelf sections at a location."""
        # Initialize monitoring for each section
        for section_id in section_ids:
            # Get the correct camera for this section
            camera_id = await self.planogram_db.get_section_camera(location_id, section_id)
            if not camera_id or camera_id not in self.camera_streams:
                print(f"No camera configured for section {section_id} at location {location_id}")
                continue

            # Start processing this camera stream if not already active
            if camera_id not in self.active_streams:
                self.active_streams[camera_id] = cv2.VideoCapture(self.camera_streams[camera_id])

            # Initialize tracking for this section
            self.last_check_times[section_id] = 0
            self.detected_issues[section_id] = []
        # endregion book:shelf-monitoring-start-monitoring

        # region book:shelf-monitoring-monitor-loop
        # Begin the monitoring loop
        while self.active_streams:
            current_time = time.time()

            # Check each section at the configured frequency
            for section_id in section_ids:
                if current_time - self.last_check_times.get(section_id, 0) >= self.check_frequency:
                    await self._check_section(location_id, section_id)
                    self.last_check_times[section_id] = current_time

            # Small delay to prevent maxing out CPU
            await asyncio.sleep(1)
        # endregion book:shelf-monitoring-monitor-loop

    # region book:shelf-monitoring-check-section
    async def _check_section(self, location_id: str, section_id: str):
        """Analyze current shelf state for a specific section."""
        # Get the correct camera and planogram
        camera_id = await self.planogram_db.get_section_camera(location_id, section_id)
        planogram = await self.planogram_db.get_section_planogram(location_id, section_id)

        if not camera_id or not planogram:
            return

        # Capture current frame
        stream = self.active_streams.get(camera_id)
        if not stream or not stream.isOpened():
            print(f"Stream not available for camera {camera_id}")
            return

        ret, frame = stream.read()
        if not ret:
            print(f"Failed to read frame from camera {camera_id}")
            return
        # endregion book:shelf-monitoring-check-section

        # region book:shelf-monitoring-check-section-process
        # Preprocess the image for the model
        input_tensor = self._preprocess_image(frame)
        # Perform object detection
        detections = self.detection_model(input_tensor)
        # Process detection results
        detected_products = self._process_detections(
            detections,
            frame.shape[1],
            frame.shape[0],
        )

        # Compare against planogram
        planogram_model = Planogram.model_validate(planogram)
        issues = self._compare_with_planogram(detected_products, planogram_model)

        # Update detected issues
        self.detected_issues[section_id] = issues
        if issues:
            timestamp = datetime.now().isoformat()

            # Report issues to inventory system for action
            await self._report_issues(location_id, section_id, issues, timestamp)
        # endregion book:shelf-monitoring-check-section-process

    # region book:shelf-monitoring-preprocess-func
    def _preprocess_image(self, image: np.ndarray) -> torch.Tensor:
        """Convert image to the format required by the model."""
        # Resize if needed
        input_size = (640, 640)  # Typical for many models
        image_resized = cv2.resize(image, input_size)

        # Convert to RGB if the image is BGR (OpenCV default)
        image_rgb = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)

        # Normalize pixel values if required by the model
        image_normalized = image_rgb / 255.0

        # Add batch dimension
        input_tensor = torch.from_numpy(image_normalized).float().unsqueeze(0)

        return input_tensor

    # endregion book:shelf-monitoring-preprocess-func

    # region book:shelf-monitoring-process-detections
    def _process_detections(
        self,
        detections,
        image_width: int,
        image_height: int,
    ) -> list[DetectedProduct]:
        """Process raw detections into structured product data."""
        detection_boxes = detections["detection_boxes"][0].detach().cpu().numpy()
        detection_classes = detections["detection_classes"][0].detach().cpu().numpy().astype(np.int32)
        detection_scores = detections["detection_scores"][0].detach().cpu().numpy()

        # Get class mappings (model-specific)
        class_mapping = self._get_class_mapping()
        # endregion book:shelf-monitoring-process-detections

        # region book:shelf-monitoring-process-detections-loop
        products: list[DetectedProduct] = []
        for i in range(len(detection_scores)):
            if detection_scores[i] >= self.confidence_threshold:
                # Convert bounding box to pixel coordinates
                box = detection_boxes[i]
                ymin, xmin, ymax, xmax = box
                box_pixel = [
                    int(ymin * image_height),
                    int(xmin * image_width),
                    int(ymax * image_height),
                    int(xmax * image_width),
                ]

                # Map class ID to product ID
                class_id = detection_classes[i]
                if class_id in class_mapping:
                    product_id = class_mapping[class_id]

                    # Store detected product info
                    products.append(
                        DetectedProduct(
                            product_id=product_id,
                            confidence=float(detection_scores[i]),
                            bounding_box=box_pixel,
                            shelf_position=ShelfPosition(
                                x=(xmin + xmax) / 2,
                                y=(ymin + ymax) / 2,
                            ),
                        )
                    )

        return products
        # endregion book:shelf-monitoring-process-detections-loop

    # region book:shelf-monitoring-class-mapping
    def _get_class_mapping(self) -> dict[int, str]:
        """Map model class IDs to product IDs."""
        # This would typically load from a configuration file
        # or database that maps between model-specific class IDs
        # and your actual retail product catalog IDs
        return {
            # Example mapping
            1: "SKU123456",  # Class 1 -> SKU123456 (Coca-Cola 12oz)
            2: "SKU789012",  # Class 2 -> SKU789012 (Pepsi 12oz)
            # ... more mappings
        }

    # endregion book:shelf-monitoring-class-mapping

    # region book:shelf-monitoring-compare-planogram
    def _compare_with_planogram(  # noqa: C901
        self,
        detected_products: list[DetectedProduct],
        planogram: Planogram,
    ) -> list[ShelfIssue]:
        """Compare detected products with expected planogram."""
        issues: list[ShelfIssue] = []

        # Group detected products by ID
        product_counts = {}
        product_positions = {}

        for product in detected_products:
            product_id = product.product_id
            if product_id in product_counts:
                product_counts[product_id] += 1
                product_positions[product_id].append(product.shelf_position)
            else:
                product_counts[product_id] = 1
                product_positions[product_id] = [product.shelf_position]

        # Check for missing products
        for expected_product in planogram.products:
            product_id = expected_product.product_id
            expected_count = expected_product.expected_count
            actual_count = product_counts.get(product_id, 0)

            if actual_count < expected_count:
                # Out of stock or low stock issue
                gap_percentage = (expected_count - actual_count) / expected_count

                issues.append(
                    ShelfIssue(
                        type="OUT_OF_STOCK" if actual_count == 0 else "LOW_STOCK",
                        product_id=product_id,
                        details={
                            "expected_count": expected_count,
                            "actual_count": actual_count,
                            "gap_percentage": gap_percentage,
                            "position": expected_product.position.model_dump(),
                        },
                    )
                )

            # Remove from counts so we can identify unexpected products
            if product_id in product_counts:
                del product_counts[product_id]

        # Any remaining products are not in the planogram
        for product_id, count in product_counts.items():
            issues.append(
                ShelfIssue(
                    type="UNEXPECTED_PRODUCT",
                    product_id=product_id,
                    details={
                        "count": count,
                        "positions": [pos.model_dump() for pos in product_positions[product_id]],
                    },
                )
            )
        # endregion book:shelf-monitoring-compare-planogram

        # region book:shelf-monitoring-compare-positions
        # Check for position issues (products in wrong places)
        for product in detected_products:
            product_id = product.product_id

            # Find this product in the planogram
            for expected_product in planogram.products:
                if expected_product.product_id == product_id:
                    # Calculate position difference
                    expected_pos = expected_product.position
                    actual_pos = product.shelf_position

                    # Calculate Euclidean distance as percentage of shelf
                    distance = np.sqrt(
                        (expected_pos.x - actual_pos.x) ** 2 + (expected_pos.y - actual_pos.y) ** 2
                    )

                    # If product is significantly out of place
                    if distance > 0.15:  # 15% of shelf dimensions
                        issues.append(
                            ShelfIssue(
                                type="MISPLACED_PRODUCT",
                                product_id=product_id,
                                details={
                                    "expected_position": expected_pos.model_dump(),
                                    "actual_position": actual_pos.model_dump(),
                                    "distance": distance,
                                },
                            )
                        )
                    break
        return issues
        # endregion book:shelf-monitoring-compare-positions

    # region book:shelf-monitoring-report-issues
    async def _report_issues(
        self,
        location_id: str,
        section_id: str,
        issues: list[ShelfIssue],
        timestamp: str,
    ):
        """Report detected issues to inventory system."""
        issue_summary = IssueSummary(
            location_id=location_id,
            section_id=section_id,
            timestamp=timestamp,
            issues=issues,
        )
        # endregion book:shelf-monitoring-report-issues

        # region book:shelf-monitoring-report-issues-send
        # Send to inventory system for processing
        await self.inventory_system.report_visual_audit(issue_summary.model_dump())

        # Log issues for monitoring
        print(f"[{timestamp}] Detected {len(issues)} issues in section {section_id} at {location_id}")
        for issue in issues:
            print(f"  - {issue.type}: {issue.product_id}")
        # endregion book:shelf-monitoring-report-issues-send
