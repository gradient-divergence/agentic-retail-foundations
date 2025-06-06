from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import heapq
from typing import Any
import numpy as np
from collections import defaultdict
from dataclasses import dataclass
from models.enums import OrderStatus

# Import models needed by these classes
from models.fulfillment import Order, Associate, Item

# --- Mock Product Data/Lookup (for planner demo) ---
# In a real system, planner would access a product database
MOCK_PRODUCT_DB = {
    "G0": {"location": (10, 10), "handling_time": 1.0, "temperature_zone": "ambient"},
    "P5": {
        "location": (25, 10),
        "handling_time": 1.2,
        "temperature_zone": "refrigerated",
    },
    "D2": {
        "location": (40, 10),
        "handling_time": 1.1,
        "temperature_zone": "refrigerated",
    },
    "F3": {"location": (10, 25), "handling_time": 1.3, "temperature_zone": "frozen"},
    "E1": {"location": (25, 25), "handling_time": 1.5, "temperature_zone": "ambient"},
    "A7": {"location": (40, 25), "handling_time": 1.4, "temperature_zone": "ambient"},
}


def get_mock_item_details(product_id: str) -> dict[str, Any]:
    # Simplified lookup, returning defaults if not found
    return MOCK_PRODUCT_DB.get(
        product_id[:2],  # Use first 2 chars for demo lookup
        {"location": (1, 1), "handling_time": 1.0, "temperature_zone": "ambient"},
    )


# ----------------------------------------------------


def calculate_remediation_timeline(steps_by_domain: dict, now_dt: datetime | None = None) -> dict:
    """Calculate a timeline for remediation steps based on estimated completion dates.

    Args:
        steps_by_domain: Dictionary mapping domain names to step details including
                         an 'estimated_completion' datetime object.
        now_dt: Optional datetime object representing the current time. Defaults to datetime.now().

    Returns:
        Dictionary containing 'critical_path', 'completion_date', 'suggested_launch_date'.
    """
    all_durations = []
    critical_path = []
    now = now_dt or datetime.now() # Use passed-in time or get current time

    for domain, step in steps_by_domain.items():
        # Ensure estimated_completion is a datetime object
        estimated_completion = step.get("estimated_completion")
        # Use datetime.datetime directly for isinstance check
        if not isinstance(estimated_completion, datetime):
            # Handle cases where it might be a string or None - skip or log error
            # For now, we'll skip if it's not a datetime
            continue

        # Calculate remaining days, ensuring non-negative duration
        duration_delta = estimated_completion - now
        duration_days = max(
            0, duration_delta.days
        )  # Use max(0, ...) to handle past dates

        all_durations.append((domain, duration_days))

    if not all_durations:
        # Handle case with no valid durations
        return {
            "critical_path": [],
            "completion_date": now,
            "suggested_launch_date": now + timedelta(days=7),  # Default suggestion
        }

    all_durations.sort(key=lambda x: x[1], reverse=True)

    # Determine critical path (e.g., top 2 longest *positive* durations)
    positive_durations = [d for d in all_durations if d[1] > 0]
    critical_path = [d[0] for d in positive_durations[:2]] # Take top 2 positive

    # Max duration determines the completion date (use original list for this)
    max_duration = all_durations[0][1] if all_durations else 0

    completion_date = now + timedelta(days=max_duration)
    suggested_launch_date = completion_date + timedelta(days=7)  # Add buffer

    return {
        "critical_path": critical_path,
        "completion_date": completion_date,
        "suggested_launch_date": suggested_launch_date,
    }


@dataclass
class StoreLayout:
    """Represents the physical layout of the store."""

    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        self.grid = np.zeros((height, width))
        self.obstacles: set[tuple[int, int]] = set()
        self.section_map: dict[tuple[int, int], str] = {}

    def add_obstacle(self, x: int, y: int):
        """Add an obstacle (e.g., shelf, wall) to the layout."""
        if 0 <= x < self.width and 0 <= y < self.height:
            self.grid[y, x] = 1  # Mark as obstacle
            self.obstacles.add((x, y))

    def add_section(
        self, top_left: tuple[int, int], bottom_right: tuple[int, int], name: str
    ):
        """Define a named section within the store."""
        x1, y1 = top_left
        x2, y2 = bottom_right
        for y in range(y1, y2 + 1):
            for x in range(x1, x2 + 1):
                if (x, y) not in self.obstacles:
                    self.section_map[(x, y)] = name

    def is_valid(self, x: int, y: int) -> bool:
        """Check if a location is within bounds and not an obstacle."""
        return 0 <= x < self.width and 0 <= y < self.height and self.grid[y, x] == 0

    def get_neighbors(self, location: tuple[int, int]) -> list[tuple[int, int]]:
        """Get valid neighboring locations (N, S, E, W)."""
        x, y = location
        neighbors = [
            (x + 1, y),
            (x - 1, y),
            (x, y + 1),
            (x, y - 1),
        ]
        return [(nx, ny) for nx, ny in neighbors if self.is_valid(nx, ny)]

    def distance(self, loc1: tuple[int, int], loc2: tuple[int, int]) -> float:
        """Calculate Manhattan distance between two locations."""
        x1, y1 = loc1
        x2, y2 = loc2
        return abs(x1 - x2) + abs(y1 - y2)

    def shortest_path(
        self, start: tuple[int, int], end: tuple[int, int]
    ) -> list[tuple[int, int]] | None:
        """Find the shortest path using A* algorithm."""
        if start == end:
            return [start]
        # Priority queue: stores tuples of (f_score, location)
        open_set: list[tuple[float, tuple[int, int]]] = []
        heapq.heappush(open_set, (0 + self.distance(start, end), start))
        # came_from maps location to the location it came from
        came_from: dict[tuple[int, int], tuple[int, int]] = {}
        # g_score maps location to the cost of the cheapest path from start
        g_score: dict[tuple[int, int], float] = {start: 0}
        # f_score maps location to g_score + heuristic (estimated total cost)
        f_score: dict[tuple[int, int], float] = {start: self.distance(start, end)}

        while open_set:
            _, current = heapq.heappop(open_set)

            if current == end:
                # Reconstruct path
                path = [current]
                while current in came_from:
                    current = came_from[current]
                    path.append(current)
                return path[::-1]  # Return reversed path

            for neighbor in self.get_neighbors(current):
                tentative_g_score = g_score[current] + 1  # Cost between neighbors is 1
                if tentative_g_score < g_score.get(neighbor, float("inf")):
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + self.distance(neighbor, end)
                    if neighbor not in [item[1] for item in open_set]:
                        heapq.heappush(open_set, (f_score[neighbor], neighbor))

        return None  # No path found


class FulfillmentPlanner:
    """Generates picking plans for orders and assigns them to associates."""

    def __init__(self, store_layout: StoreLayout):
        self.store_layout = store_layout
        self.orders: list[Order] = []
        self.associates: list[Associate] = []
        self.assignments: dict[str, list[Order]] = defaultdict(
            list
        )  # associate_id -> list of assigned orders
        self.picking_paths: dict[
            str, list[tuple[int, int]]
        ] = {}  # associate_id -> path
        self.estimated_times: dict[str, float] = {}  # associate_id -> total time

    def add_order(self, order: Order):
        """Add a customer order to the planner."""
        self.orders.append(order)

    def add_associate(self, associate: Associate):
        """Add an available store associate."""
        self.associates.append(associate)

    def plan(self):
        """Create the fulfillment plan: batch orders, assign, find paths."""
        # 1. Prioritize Orders (Sort by created_at instead of missing due_time)
        self.orders.sort(key=lambda o: o.created_at)

        # 2. Simple Assignment
        unassigned_orders = self.orders[:] # Copy list
        self.assignments = defaultdict(list)
        self.picking_paths = {}
        self.estimated_times = {}

        for associate in self.associates:
            current_time = 0.0
            current_location = associate.current_location
            assigned_to_associate: list[Order] = []
            path_for_associate: list[tuple[int, int]] = [current_location]

            # Try assigning orders to this associate
            temp_unassigned = unassigned_orders[:]
            for order in temp_unassigned:
                # Get required zones from item details via lookup
                required_zones = set()
                items_details = []
                valid_order = True
                for line_item in order.items:
                    item_detail = get_mock_item_details(line_item.product_id)
                    if not item_detail:  # Handle if product not found
                        print(
                            f"Warning: Details not found for product {line_item.product_id} in order {order.order_id}"
                        )
                        valid_order = False
                        break
                    required_zones.add(item_detail["temperature_zone"])
                    items_details.append(item_detail)  # Store for later use

                if not valid_order:
                    continue  # Skip order if item details missing

                can_handle = all(
                    zone in associate.authorized_zones for zone in required_zones
                )
                if not can_handle:
                    continue

                # Estimate time for this order using looked-up details
                order_path, order_time = self._estimate_order_time(
                    order, items_details, current_location, associate.efficiency
                )
                if order_path is None:
                    continue  # Cannot find path

                # Check if associate has time before shift ends
                if (
                    associate.shift_end_time is not None
                    and (current_time + order_time) > associate.shift_end_time
                ):
                    continue  # Not enough time

                # Assign order if checks pass
                assigned_to_associate.append(order)
                current_time += order_time
                current_location = order_path[
                    -1
                ]  # End location of this order becomes start for next
                path_for_associate.extend(
                    order_path[1:]
                )  # Append path, skip start node
                unassigned_orders.remove(order)

            if assigned_to_associate:
                self.assignments[associate.associate_id] = assigned_to_associate
                self.picking_paths[associate.associate_id] = path_for_associate
                self.estimated_times[associate.associate_id] = current_time
                # Update associate's status if needed (e.g., busy until estimated_time)

        # Update status of assigned/unassigned orders
        for order in self.orders:
            is_assigned = any(
                order in assigned_list for assigned_list in self.assignments.values()
            )
            # Use OrderStatus enum correctly
            order.status = OrderStatus.ALLOCATED if is_assigned else OrderStatus.CREATED

    def _estimate_order_time(
        self,
        order: Order,
        item_details: list[dict[str, Any]],
        start_location: tuple[int, int],
        efficiency: float,
    ) -> tuple[list[tuple[int, int]] | None, float]:
        """Estimate the time to pick all items in an order starting from a location."""
        total_time = 0.0
        current_location = start_location
        full_path: list[tuple[int, int]] = [start_location]
        # Get locations from the passed item_details
        pick_locations = [detail["location"] for detail in item_details]

        # Simple nearest neighbor heuristic for picking order
        remaining_locations = pick_locations[:]
        while remaining_locations:
            nearest_loc = min(
                remaining_locations,
                key=lambda loc: self.store_layout.distance(current_location, loc),
            )
            segment_path = self.store_layout.shortest_path(
                current_location, nearest_loc
            )
            if segment_path is None:
                return None, float("inf")  # Cannot reach item

            travel_time = (len(segment_path) - 1) * 0.1  # Example: 0.1 min per step
            # Get handling time from the passed item_details matching the location
            handling_time = sum(
                detail["handling_time"]
                for detail in item_details
                if detail["location"] == nearest_loc
            )
            total_time += (travel_time + handling_time) / efficiency
            full_path.extend(segment_path[1:])
            current_location = nearest_loc
            remaining_locations.remove(nearest_loc)

        # Add time to return to a drop-off point
        return_path = self.store_layout.shortest_path(current_location, start_location)
        if return_path:
            travel_time = (len(return_path) - 1) * 0.1
            total_time += travel_time / efficiency
            full_path.extend(return_path[1:])

        return full_path, total_time

    def explain_plan(self) -> str:
        """Generate a human-readable explanation of the plan."""
        explanation: list[str] = ["Fulfillment Plan Summary:"]
        total_orders = len(self.orders)
        assigned_count = sum(len(orders) for orders in self.assignments.values())
        unassigned_count = total_orders - assigned_count

        explanation.append(f"- Total Orders: {total_orders}")
        explanation.append(f"- Assigned Orders: {assigned_count}")
        explanation.append(f"- Unassigned Orders: {unassigned_count}")

        explanation.append("\nAssignments Details:")
        for associate_id, assigned_orders in self.assignments.items():
            # Add back ignore for persistent assignment error
            associate = next(  # type: ignore[assignment]
                (a for a in self.associates if a.associate_id == associate_id), None
            )
            if not associate:
                assoc_name = f"Associate {associate_id}"
            else:
                assoc_name = associate.name
            explanation.append(
                f"\n- {assoc_name} ({associate_id}): {len(assigned_orders)} orders, Est. Time: {self.estimated_times.get(associate_id, 0):.1f} min"
            )
            for order in assigned_orders:
                # Remove priority from explanation
                explanation.append(
                    f"  - Order {order.order_id} (Items: {len(order.items)})"
                )

        if unassigned_count > 0:
            explanation.append("\nUnassigned Orders:")
            for order in self.orders:
                # Check order status enum value
                if order.status == OrderStatus.CREATED:
                    # Remove priority from explanation
                    explanation.append(
                        f"  - Order {order.order_id} (Items: {len(order.items)})"
                    )

        return "\n".join(explanation)

    def visualize_plan(self) -> plt.Figure | None:
        """Visualize the store layout and picking paths."""
        try:
            fig, ax = plt.subplots(figsize=(12, 10))
            # Draw grid
            ax.imshow(self.store_layout.grid, cmap="Greys", origin="lower", alpha=0.3)
            # Draw sections
            section_colors = plt.cm.get_cmap(
                "tab20", len(self.store_layout.section_map)
            )
            unique_sections = sorted(list(set(self.store_layout.section_map.values())))
            color_map = {
                name: section_colors(i) for i, name in enumerate(unique_sections)
            }
            for (x, y), name in self.store_layout.section_map.items():
                ax.add_patch(
                    plt.Rectangle(
                        (x - 0.5, y - 0.5), 1, 1, color=color_map[name], alpha=0.2
                    )
                )
            # Draw obstacles
            for x, y in self.store_layout.obstacles:
                ax.add_patch(plt.Rectangle((x - 0.5, y - 0.5), 1, 1, color="black"))
            # Draw paths
            path_colors = plt.cm.get_cmap("viridis", len(self.picking_paths))
            for i, (assoc_id, path) in enumerate(self.picking_paths.items()):
                if not path:
                    continue
                path_x, path_y = zip(*path)
                ax.plot(
                    path_x,
                    path_y,
                    marker=".",
                    linestyle="-",
                    label=f"Assoc {assoc_id}",
                    color=path_colors(i),
                    linewidth=2,
                )
                ax.plot(
                    path_x[0], path_y[0], "go", markersize=8, label=f"{assoc_id} Start"
                )  # Start point
                ax.plot(
                    path_x[-1], path_y[-1], "ro", markersize=8, label=f"{assoc_id} End"
                )  # End point

            ax.set_xticks(np.arange(-0.5, self.store_layout.width, 5))
            ax.set_yticks(np.arange(-0.5, self.store_layout.height, 5))
            ax.set_xticklabels(np.arange(0, self.store_layout.width + 1, 5))
            ax.set_yticklabels(np.arange(0, self.store_layout.height + 1, 5))
            ax.grid(True, which="major", linestyle=":", linewidth="0.5", color="gray")
            ax.set_xlim(-0.5, self.store_layout.width - 0.5)
            ax.set_ylim(-0.5, self.store_layout.height - 0.5)
            ax.set_title("Store Layout and Fulfillment Paths")
            ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
            plt.tight_layout(rect=(0, 0, 0.85, 1))  # Use tuple for rect
            return fig
        except Exception as e:
            print(f"Error during visualization: {e}")
            return None
