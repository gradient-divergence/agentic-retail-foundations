"""
Planning agents and related classes for store fulfillment optimization.
Includes Associate, StoreLayout, and FulfillmentPlanner.
"""

import numpy as np
import matplotlib.pyplot as plt
import heapq
from typing import Optional, Dict, List, Set, Tuple

# Import models needed by these classes
from models.fulfillment import Order


class Associate:
    """Represents a store associate who can fulfill orders."""

    def __init__(
        self,
        associate_id: str,
        name: str,
        efficiency: float = 1.0,
        authorized_zones: Optional[list[str]] = None,
        current_location: tuple[int, int] = (0, 0),
        shift_end_time: float | None = None,
    ):
        self.associate_id = associate_id
        self.name = name
        self.efficiency = efficiency  # multiplier for picking speed
        self.authorized_zones = authorized_zones if authorized_zones is not None else [
            "ambient",
            "refrigerated",
            "frozen",
        ]
        self.current_location = current_location
        self.shift_end_time = shift_end_time  # minutes from now
        self.assigned_orders: list[Order] = []  # Explicitly type hint
        self.status = "available"  # available, busy

    def can_handle_order(self, order: Order) -> bool:
        """Check if associate is authorized for all temperature zones in order."""
        return all(
            zone in self.authorized_zones for zone in order.get_temperature_zones()
        )

    def estimate_time_to_complete(self, orders: list[Order]) -> float:
        """Estimate time to complete a list of orders."""
        return sum(order.estimate_picking_time(self.efficiency) for order in orders)

    def available_time(self) -> float:  # Return float, not Optional
        """Return the available time in minutes before shift ends."""
        if self.shift_end_time is None:
            return float("inf")
        # Assuming current time is 0 for relative calculation
        return max(0.0, self.shift_end_time)

    def __repr__(self):
        return (
            f"Associate({self.associate_id}: {self.name}, efficiency {self.efficiency})"
        )


class StoreLayout:
    """Represents the physical layout of the store."""

    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        self.grid = np.zeros((height, width))
        self.obstacles: Set[Tuple[int, int]] = set()
        self.section_map: Dict[Tuple[int, int], str] = {}

    def add_obstacle(self, x: int, y: int):
        """Mark a location as an obstacle."""
        if 0 <= x < self.width and 0 <= y < self.height:
            self.obstacles.add((x, y))
            self.grid[y, x] = 1
        else:
            print(
                f"Warning: Obstacle ({x},{y}) outside store bounds ({self.width}x{self.height})"
            )

    def add_section(
        self, x_range: tuple[int, int], y_range: tuple[int, int], section_name: str
    ):
        """Define a named section of the store."""
        for x in range(x_range[0], x_range[1] + 1):
            for y in range(y_range[0], y_range[1] + 1):
                if 0 <= x < self.width and 0 <= y < self.height:
                    self.section_map[(x, y)] = section_name
                else:
                    print(
                        f"Warning: Section coordinate ({x},{y}) outside store bounds."
                    )

    def get_section(self, location: tuple[int, int]) -> str:
        """Get the section name for a location."""
        return self.section_map.get(location, "unknown")

    def distance(self, loc1: tuple[int, int], loc2: tuple[int, int]) -> float:
        """Calculate Manhattan distance."""
        return abs(loc1[0] - loc2[0]) + abs(loc1[1] - loc2[1])

    def shortest_path(
        self, start: tuple[int, int], end: tuple[int, int]
    ) -> list[tuple[int, int]]:
        """Find shortest path using A*."""
        if start == end:
            return [start]
        open_set: List[Tuple[float, Tuple[int, int]]] = []
        heapq.heappush(open_set, (0 + self.distance(start, end), start))
        came_from: Dict[Tuple[int, int], Tuple[int, int]] = {}
        g_score: Dict[Tuple[int, int], float] = {start: 0}
        f_score: Dict[Tuple[int, int], float] = {start: self.distance(start, end)}

        while open_set:
            current_f, current = heapq.heappop(open_set)
            if current == end:
                path = [current]
                while current in came_from:
                    current = came_from[current]
                    path.append(current)
                return path[::-1]

            for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                neighbor = (current[0] + dx, current[1] + dy)
                if (
                    0 <= neighbor[0] < self.width
                    and 0 <= neighbor[1] < self.height
                    and neighbor not in self.obstacles
                ):
                    tentative_g = g_score[current] + 1
                    if neighbor not in g_score or tentative_g < g_score[neighbor]:
                        came_from[neighbor] = current
                        g_score[neighbor] = tentative_g
                        f_score[neighbor] = tentative_g + self.distance(neighbor, end)
                        heapq.heappush(open_set, (f_score[neighbor], neighbor))
        return []  # No path found

    def optimize_path(
        self, locations: list[tuple[int, int]], start: tuple[int, int]
    ) -> list[tuple[int, int]]:
        """Optimize picking path using nearest-neighbor."""
        if not locations:
            return [start]
        current = start
        unvisited = set(locations)
        path = [current]
        while unvisited:
            nearest = min(unvisited, key=lambda loc: self.distance(current, loc))
            unvisited.remove(nearest)
            path.append(nearest)
            current = nearest
        return path

    def visualize(self, item_locations=None, associate_locations=None, paths=None):
        """Visualize the store layout."""
        fig, ax = plt.subplots(figsize=(self.width / 5, self.height / 5))
        ax.imshow(self.grid, cmap="Greys", alpha=0.3, origin="lower")

        # Plot section boundaries/labels (simplified)
        section_colors = plt.cm.get_cmap("tab10", len(self.section_map))
        color_map = {
            name: section_colors(i)
            for i, name in enumerate(set(self.section_map.values()))
        }
        for loc, name in self.section_map.items():
            ax.scatter(loc[0], loc[1], color=color_map[name], alpha=0.1, s=50)

        # Plot items
        if item_locations:
            xs = [loc[0] for loc in item_locations]
            ys = [loc[1] for loc in item_locations]
            ax.scatter(xs, ys, color="blue", marker="s", label="Items", s=30, alpha=0.8)

        # Plot associates
        if associate_locations:
            xs = [loc[0] for loc in associate_locations]
            ys = [loc[1] for loc in associate_locations]
            ax.scatter(
                xs,
                ys,
                color="red",
                marker="^",
                s=100,
                label="Associates",
                edgecolors="black",
            )

        # Plot paths
        if paths:
            path_colors = plt.cm.get_cmap("viridis", len(paths))
            for i, path in enumerate(paths):
                if len(path) > 1:
                    xs = [loc[0] for loc in path]
                    ys = [loc[1] for loc in path]
                    ax.plot(
                        xs,
                        ys,
                        marker=".",
                        linestyle="-",
                        color=path_colors(i),
                        alpha=0.7,
                        linewidth=2,
                        label=f"Path {i + 1}",
                    )

        ax.set_xticks(np.arange(0, self.width, 5))
        ax.set_yticks(np.arange(0, self.height, 5))
        ax.set_xlabel("X Coordinate")
        ax.set_ylabel("Y Coordinate")
        ax.set_title("Store Layout with Fulfillment Plan")
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.15), ncol=3)
        ax.grid(True, linestyle=":", alpha=0.5)
        plt.tight_layout()
        return fig  # Return the figure object


class FulfillmentPlanner:
    """Plans and optimizes order fulfillment in a retail store."""

    def __init__(self, store_layout: StoreLayout):
        self.store_layout = store_layout
        self.orders: list[Order] = []
        self.associates: list[Associate] = []
        self.assignments: dict[str, list[Order]] = {}
        self.paths: dict[str, list[tuple[int, int]]] = {}

    def add_order(self, order: Order):
        self.orders.append(order)

    def add_associate(self, associate: Associate):
        self.associates.append(associate)

    def batch_orders(self, max_items_per_batch: int = 10) -> list[list[Order]]:
        """Group orders into batches."""
        pending_orders = [o for o in self.orders if o.status == "pending"]
        sorted_orders = sorted(
            pending_orders,
            key=lambda o: (
                -o.priority,
                o.due_time if o.due_time is not None else float("inf"),
            ),
        )
        batches = []
        used_order_ids = set()

        for i, order in enumerate(sorted_orders):
            if order.order_id in used_order_ids:
                continue

            current_batch = [order]
            current_items = len(order.items)
            used_order_ids.add(order.order_id)

            # Try to add more orders to this batch
            for j in range(i + 1, len(sorted_orders)):
                next_order = sorted_orders[j]
                if next_order.order_id in used_order_ids:
                    continue
                if current_items + len(next_order.items) <= max_items_per_batch:
                    # Simple proximity check (optional, can add complexity)
                    # if self._batches_compatible(current_batch, next_order):
                    current_batch.append(next_order)
                    current_items += len(next_order.items)
                    used_order_ids.add(next_order.order_id)

            batches.append(current_batch)
        return batches

    def optimize_assignments(self) -> tuple[dict[str, list[Order]], list[Order]]:
        """Assign order batches to associates."""
        self.assignments = {a.associate_id: [] for a in self.associates}
        unassigned_orders = []
        available_associates = [a for a in self.associates if a.status == "available"]
        sorted_associates = sorted(available_associates, key=lambda a: -a.efficiency)
        batches = self.batch_orders()

        for batch in batches:
            best_fit = None
            min_cost = float("inf")

            for associate in sorted_associates:
                if not all(associate.can_handle_order(o) for o in batch):
                    continue

                # Estimate travel time + picking time
                locations = set()
                for o in batch:
                    locations.update(o.get_item_locations())
                optimized_path = self.store_layout.optimize_path(
                    list(locations), associate.current_location
                )
                travel_distance = sum(
                    self.store_layout.distance(optimized_path[k], optimized_path[k + 1])
                    for k in range(len(optimized_path) - 1)
                )
                avg_speed = 30  # units per minute
                travel_time = travel_distance / avg_speed
                picking_time = associate.estimate_time_to_complete(batch)
                total_time = travel_time + picking_time

                # Check if associate has enough time
                if associate.available_time() < total_time:
                    continue

                # Simple cost: time taken (can be improved)
                cost = total_time
                if cost < min_cost:
                    min_cost = cost
                    best_fit = associate

            if best_fit:
                self.assignments[best_fit.associate_id].extend(batch)
                for order in batch:
                    order.assigned_to = best_fit.associate_id
                    order.status = "assigned"
            else:
                unassigned_orders.extend(batch)
                for order in batch:
                    order.status = "unassigned"

        return self.assignments, unassigned_orders

    def generate_picking_paths(self):
        """Generate optimized paths for assigned batches."""
        self.paths = {}
        for associate_id, assigned_orders in self.assignments.items():
            if not assigned_orders:
                continue
            associate = next(
                (a for a in self.associates if a.associate_id == associate_id), None
            )
            if not associate:
                continue

            all_locations = set()
            for order in assigned_orders:
                all_locations.update(order.get_item_locations())

            if all_locations:
                optimized_path = self.store_layout.optimize_path(
                    list(all_locations), associate.current_location
                )
                self.paths[associate_id] = optimized_path
            else:
                self.paths[associate_id] = [associate.current_location]

    def plan(self) -> dict:
        """Generate the complete fulfillment plan."""
        assignments, unassigned = self.optimize_assignments()
        self.generate_picking_paths()
        return {
            "assignments": assignments,
            "paths": self.paths,
            "unassigned_orders": unassigned,
        }

    def visualize_plan(self):
        """Visualize the generated fulfillment plan."""
        item_locations = set()
        for orders in self.assignments.values():
            for order in orders:
                item_locations.update(order.get_item_locations())

        associate_locations = [
            a.current_location
            for a in self.associates
            if a.associate_id in self.assignments and self.assignments[a.associate_id]
        ]
        paths = [
            self.paths[a_id] for a_id in self.assignments if self.assignments[a_id]
        ]
        return self.store_layout.visualize(
            item_locations=list(item_locations),
            associate_locations=associate_locations,
            paths=paths,
        )

    def explain_plan(self) -> str:
        """Generate a human-readable explanation."""
        explanation: List[str] = ["Fulfillment Plan Summary:"]
        total_orders = len(self.orders)
        assigned_count = sum(len(orders) for orders in self.assignments.values())
        unassigned_count = total_orders - assigned_count

        explanation.append(f"- Total Orders: {total_orders}")
        explanation.append(f"- Assigned Orders: {assigned_count}")
        explanation.append(f"- Unassigned Orders: {unassigned_count}")
        explanation.append(
            f"- Associates Utilized: {len([a for a in self.assignments if self.assignments[a]])}"
        )

        explanation.append("\nAssignments Details:")
        for associate_id, assigned_orders in self.assignments.items():
            if not assigned_orders:
                continue
            # Add back ignore for persistent assignment error
            associate = next( # type: ignore[assignment]
                (a for a in self.associates if a.associate_id == associate_id), None
            )
            if not associate:
                continue

            path = self.paths.get(associate_id, [])
            total_distance = (
                sum(
                    self.store_layout.distance(path[i], path[i + 1])
                    for i in range(len(path) - 1)
                )
                if len(path) > 1
                else 0
            )
            num_items = sum(len(o.items) for o in assigned_orders)
            est_time = associate.estimate_time_to_complete(assigned_orders)
            # Add estimated travel time
            avg_speed = 30
            travel_time = total_distance / avg_speed if avg_speed > 0 else 0
            total_est_time = est_time + travel_time

            explanation.append(f"\n -> {associate.name} ({associate_id}):")
            explanation.append(f"    - Orders Assigned: {len(assigned_orders)}")
            explanation.append(f"    - Total Items: {num_items}")
            explanation.append(f"    - Est. Picking Time: {est_time:.1f} min")
            explanation.append(
                f"    - Est. Travel Distance: {total_distance:.1f} units"
            )
            explanation.append(
                f"    - Est. Total Time: {total_est_time:.1f} min (Available: {associate.available_time():.1f} min)"
            )
            explanation.append(
                f"    - Temp Zones: {', '.join(set.union(*[o.get_temperature_zones() for o in assigned_orders]))}"
            )

        if unassigned_count > 0:
            explanation.append("\nUnassigned Orders:")
            for order in self.orders:
                if order.status == "unassigned":
                    explanation.append(
                        f"  - {order.order_id} (Priority: {order.priority}, Items: {len(order.items)})"
                    )

        return "\n".join(explanation)
