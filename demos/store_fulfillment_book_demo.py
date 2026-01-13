# region book:store-fulfillment-imports
import heapq
import random
import time
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np


# Represents a single product within the store's inventory,
# including its location and handling characteristics.
class Item:
    """Represents a product in the store inventory."""

    def __init__(
        self,
        item_id: str,
        name: str,
        category: str,
        location: tuple[int, int],
        temperature_zone: str = "ambient",
        handling_time: float = 1.0,
        fragility: float = 0.0,
    ):
        self.item_id = item_id
        self.name = name
        self.category = category
        self.location = location  # (x, y) coordinates in store
        self.temperature_zone = temperature_zone  # "ambient", "refrigerated", "frozen"
        self.handling_time = handling_time  # base time to pick in minutes
        self.fragility = fragility  # 0.0 to 1.0, affects stacking and handling

    def __repr__(self):
        return f"Item({self.item_id}: {self.name} at {self.location})"


# endregion book:store-fulfillment-imports


# region book:store-fulfillment-order-class
# Represents a customer's request, containing multiple items and associated
# constraints like priority and due time.
class Order:
    """Represents a customer order with multiple items."""

    def __init__(
        self,
        order_id: str,
        items: list[Item],
        priority: int = 1,
        due_time: float | None = None,
    ):
        self.order_id = order_id
        self.items = items
        self.priority = priority  # 1 (standard) to 5 (highest)
        self.due_time = due_time  # minutes from now
        self.assigned_to = None
        self.status = "pending"  # pending, in_progress, completed

    def get_temperature_zones(self) -> set[str]:
        """Return the set of temperature zones required for this order."""
        return {item.temperature_zone for item in self.items}

    def get_item_locations(self) -> list[tuple[int, int]]:
        """Return the locations of all items in the order."""
        return [item.location for item in self.items]

    def estimate_picking_time(self, associate_efficiency: float = 1.0) -> float:
        """Estimate the time to pick all items in the order."""
        # Base handling time for all items
        base_time = sum(item.handling_time for item in self.items)
        # Adjust for associate efficiency
        return base_time / associate_efficiency

    def __repr__(self):
        return f"Order({self.order_id}: {len(self.items)} items, priority {self.priority})"


# endregion book:store-fulfillment-order-class


# region book:store-fulfillment-associate-class
# Models the store personnel responsible for picking orders, including their
# efficiency, authorized work zones, and availability.
class Associate:
    """Represents a store associate who can fulfill orders."""

    def __init__(
        self,
        associate_id: str,
        name: str,
        efficiency: float = 1.0,
        authorized_zones: list[str] = None,
        current_location: tuple[int, int] = (0, 0),
        shift_end_time: float | None = None,
    ):
        self.associate_id = associate_id
        self.name = name
        self.efficiency = efficiency  # multiplier for picking speed
        self.authorized_zones = authorized_zones or [
            "ambient",
            "refrigerated",
            "frozen",
        ]
        self.current_location = current_location
        self.shift_end_time = shift_end_time  # minutes from now
        self.assigned_orders = []
        self.status = "available"  # available, busy

    # endregion book:store-fulfillment-associate-class

    # region book:store-fulfillment-associate-methods
    def can_handle_order(self, order: Order) -> bool:
        """Check if associate is authorized for all temperature zones in order."""
        return all(zone in self.authorized_zones for zone in order.get_temperature_zones())

    def estimate_time_to_complete(self, orders: list[Order]) -> float:
        """Estimate time to complete a list of orders."""
        return sum(order.estimate_picking_time(self.efficiency) for order in orders)

    def available_time(self) -> float | None:
        """Return the available time in minutes before shift ends."""
        if self.shift_end_time is None:
            return float("inf")
        return max(0, self.shift_end_time)

    def __repr__(self):
        return f"Associate({self.associate_id}: {self.name}, efficiency {self.efficiency})"

    # endregion book:store-fulfillment-associate-methods


# region book:store-fulfillment-layout-class
# Models the store's physical grid, including obstacles and section definitions, crucial for pathfinding.
class StoreLayout:
    """Represents the physical layout of the store."""

    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        self.grid = np.zeros((height, width))
        self.obstacles = set()  # (x, y) coordinates of obstacles
        self.section_map = {}  # maps (x, y) to section name

    def add_obstacle(self, x: int, y: int):
        """Mark a location as an obstacle (cannot be traversed)."""
        self.obstacles.add((x, y))
        self.grid[y, x] = 1

    def add_section(self, x_range: tuple[int, int], y_range: tuple[int, int], section_name: str):
        """Define a named section of the store."""
        for x in range(x_range[0], x_range[1] + 1):
            for y in range(y_range[0], y_range[1] + 1):
                self.section_map[(x, y)] = section_name

    # endregion book:store-fulfillment-layout-class

    # region book:store-fulfillment-layout-helpers
    def get_section(self, location: tuple[int, int]) -> str:
        """Get the section name for a location."""
        return self.section_map.get(location, "unknown")

    def distance(self, loc1: tuple[int, int], loc2: tuple[int, int]) -> float:
        """Calculate Manhattan distance between two locations."""
        return abs(loc1[0] - loc2[0]) + abs(loc1[1] - loc2[1])

    def shortest_path(self, start: tuple[int, int], end: tuple[int, int]) -> list[tuple[int, int]]:
        """Find shortest path between two points using A* algorithm."""
        if start == end:
            return [start]

        # A* algorithm (pathfinding to navigate around obstacles)
        open_set = []
        heapq.heappush(open_set, (0, start))
        came_from = {}
        g_score = {start: 0}
        f_score = {start: self.distance(start, end)}

        while open_set:
            _, current = heapq.heappop(open_set)

            if current == end:
                # Reconstruct path
                path = [current]
                while current in came_from:
                    current = came_from[current]
                    path.append(current)
                return path[::-1]

            for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                neighbor = (current[0] + dx, current[1] + dy)
                # Check bounds and obstacles
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

        # No path found
        return []

    # endregion book:store-fulfillment-layout-helpers

    # region book:store-fulfillment-optimize-path
    def optimize_path(
        self, locations: list[tuple[int, int]], start: tuple[int, int]
    ) -> list[tuple[int, int]]:
        """Optimize picking path using a greedy nearest-neighbor approach."""
        if not locations:
            return []

        current = start
        unvisited = set(locations)
        path = [current]

        while unvisited:
            # Find nearest unvisited location
            nearest = min(unvisited, key=lambda loc: self.distance(current, loc))
            current = nearest
            path.append(current)
            unvisited.remove(nearest)

        return path

    # endregion book:store-fulfillment-optimize-path

    # region book:store-fulfillment-visualize
    def visualize(self, item_locations=None, associate_locations=None, paths=None):
        """Visualize the store layout with items, associates and paths."""
        plt.figure(figsize=(10, 8))
        # Plot store grid
        plt.imshow(self.grid, cmap="Greys", alpha=0.3)
        # Plot section boundaries
        sections = defaultdict(list)
        for (x, y), section in self.section_map.items():
            sections[section].append((x, y))

        for section, points in sections.items():
            xs = [p[0] for p in points]
            ys = [p[1] for p in points]
            plt.scatter(xs, ys, alpha=0.2, label=section)
        # endregion book:store-fulfillment-visualize

        # region book:store-fulfillment-visualize-items
        # Plot items
        if item_locations:
            xs = [loc[0] for loc in item_locations]
            ys = [loc[1] for loc in item_locations]
            plt.scatter(xs, ys, color="blue", marker="s", label="Items")

        # Plot associates
        if associate_locations:
            xs = [loc[0] for loc in associate_locations]
            ys = [loc[1] for loc in associate_locations]
            plt.scatter(xs, ys, color="red", marker="^", s=100, label="Associates")

        # Plot paths
        if paths:
            for i, path in enumerate(paths):
                xs = [loc[0] for loc in path]
                ys = [loc[1] for loc in path]
                plt.plot(xs, ys, "g-", alpha=0.7, label=f"Path {i + 1}" if i == 0 else "")

        plt.legend(loc="upper center", bbox_to_anchor=(0.5, 1.1), ncol=3)
        plt.title("Store Layout with Fulfillment Plan")
        plt.tight_layout()
        plt.show()
        # endregion book:store-fulfillment-visualize-items


# region book:store-fulfillment-planner-class
# The core planning engine that takes orders, associates, and the store layout
# to generate optimized assignments and picking paths.
class FulfillmentPlanner:
    """Plans and optimizes order fulfillment in a retail store."""

    def __init__(self, store_layout: StoreLayout):
        self.store_layout = store_layout
        self.orders = []
        self.associates = []
        self.assignments = {}  # associate_id -> [orders]
        self.paths = {}  # associate_id -> path

    def add_order(self, order: Order):
        """Add an order to be fulfilled."""
        self.orders.append(order)

    def add_associate(self, associate: Associate):
        """Add an associate available for fulfillment."""
        self.associates.append(associate)

    # endregion book:store-fulfillment-planner-class

    # region book:store-fulfillment-batch-orders
    def batch_orders(self, max_items_per_batch: int = 10) -> list[list[Order]]:
        """Group orders into batches for efficient picking."""
        # Sort orders by priority (highest first)
        sorted_orders = sorted(self.orders, key=lambda o: -o.priority)
        batches = []
        current_batch = []
        current_items = 0

        for order in sorted_orders:
            # If adding this order would exceed the max items, start a new batch
            if current_items + len(order.items) > max_items_per_batch and current_batch:
                batches.append(current_batch)
                current_batch = []
                current_items = 0

            current_batch.append(order)
            current_items += len(order.items)

        # Add the last batch if not empty
        if current_batch:
            batches.append(current_batch)

        return batches

    # endregion book:store-fulfillment-batch-orders

    # region book:store-fulfillment-optimize-assignments
    def optimize_assignments(self):
        """Assign orders to associates optimally."""
        # Reset assignments
        self.assignments = {a.associate_id: [] for a in self.associates}
        # Group orders into batches
        batches = self.batch_orders()
        # Sort associates by efficiency (highest first)
        sorted_associates = sorted(self.associates, key=lambda a: -a.efficiency)
        # Assign batches to associates
        for batch in batches:
            # Find the best associate for this batch
            best_associate = None
            min_completion_time = float("inf")

            for associate in sorted_associates:
                # Check if associate can handle all orders in batch
                if not all(associate.can_handle_order(order) for order in batch):
                    continue

                # Calculate estimated completion time
                current_workload = associate.estimate_time_to_complete(
                    self.assignments.get(associate.associate_id, [])
                )
                batch_time = associate.estimate_time_to_complete(batch)
                total_time = current_workload + batch_time

                # Check if associate has enough time in shift
                if associate.available_time() < total_time:
                    continue

                if total_time < min_completion_time:
                    min_completion_time = total_time
                    best_associate = associate
            # endregion book:store-fulfillment-optimize-assignments

            # region book:store-fulfillment-assign-batch
            # Assign batch to best associate or leave unassigned
            if best_associate:
                self.assignments[best_associate.associate_id].extend(batch)
                for order in batch:
                    order.assigned_to = best_associate.associate_id
            else:
                # Could not assign this batch
                for order in batch:
                    order.status = "unassigned"
            # endregion book:store-fulfillment-assign-batch

    # region book:store-fulfillment-generate-paths
    def generate_picking_paths(self):
        """Generate optimized picking paths for each associate."""
        self.paths = {}

        for associate in self.associates:
            assigned_orders = self.assignments.get(associate.associate_id, [])
            if not assigned_orders:
                continue

            # Collect all item locations from assigned orders
            all_locations = []
            for order in assigned_orders:
                all_locations.extend(order.get_item_locations())

            # Optimize path starting from associate's current location
            optimized_path = self.store_layout.optimize_path(all_locations, associate.current_location)
            self.paths[associate.associate_id] = optimized_path

    # endregion book:store-fulfillment-generate-paths

    # region book:store-fulfillment-plan
    def plan(self):
        """Generate a complete fulfillment plan."""
        self.optimize_assignments()
        self.generate_picking_paths()
        # Return summary of plan
        return {
            "assignments": self.assignments,
            "paths": self.paths,
            "unassigned": [o for o in self.orders if o.status == "unassigned"],
        }

    # endregion book:store-fulfillment-plan

    # region book:store-fulfillment-visualize-plan
    def visualize_plan(self):
        """Visualize the fulfillment plan."""
        # Collect all item locations
        item_locations = []
        for order in self.orders:
            if order.status != "unassigned":
                item_locations.extend(order.get_item_locations())

        # Collect associate locations and paths
        associate_locations = [a.current_location for a in self.associates]
        paths = list(self.paths.values())
        # Visualize
        self.store_layout.visualize(
            item_locations=item_locations,
            associate_locations=associate_locations,
            paths=paths,
        )

    # endregion book:store-fulfillment-visualize-plan

    # region book:store-fulfillment-explain-plan
    def explain_plan(self) -> str:
        """Generate a human-readable explanation of the fulfillment plan."""
        explanation = []
        explanation.append("Fulfillment Plan Summary:")
        explanation.append(f"- Total orders: {len(self.orders)}")
        explanation.append(f"- Available associates: {len(self.associates)}")
        assigned_count = sum(1 for o in self.orders if o.status != "unassigned")
        explanation.append(f"- Orders assigned: {assigned_count}")
        explanation.append(f"- Orders unassigned: {len(self.orders) - assigned_count}")
        explanation.append("\nAssignments:")
        for associate in self.associates:
            assigned = self.assignments.get(associate.associate_id, [])
            if assigned:
                path = self.paths.get(associate.associate_id, [])
                total_distance = (
                    sum(self.store_layout.distance(path[i], path[i + 1]) for i in range(len(path) - 1))
                    if len(path) > 1
                    else 0
                )
                explanation.append(f"\n{associate.name}:")
                explanation.append(f"- Orders: {len(assigned)}")
                explanation.append(f"- Items: {sum(len(o.items) for o in assigned)}")
                explanation.append(
                    f"- Estimated time: {associate.estimate_time_to_complete(assigned):.1f} minutes"
                )
                explanation.append(f"- Walking distance: {total_distance} units")
                zones = set().union(*(o.get_temperature_zones() for o in assigned))
                explanation.append(f"- Temperature zones: {', '.join(zones)}")

        return "\n".join(explanation)

    # endregion book:store-fulfillment-explain-plan


# region book:store-fulfillment-demo-setup
# Example usage
def demo_fulfillment_system():  # noqa: C901
    """Demonstrate the fulfillment optimization system with a sample scenario."""
    # Create store layout
    store = StoreLayout(width=50, height=40)
    store.add_section((5, 15), (5, 15), "Grocery")
    store.add_section((20, 30), (5, 15), "Produce")
    store.add_section((35, 45), (5, 15), "Dairy")
    store.add_section((5, 15), (20, 30), "Frozen")
    store.add_section((20, 30), (20, 30), "Electronics")
    store.add_section((35, 45), (20, 30), "Apparel")
    # Add obstacles (walls, displays, etc.)
    for x in range(0, 50, 10):
        for y in range(0, 40):
            if y % 5 != 0:  # Leave gaps for aisles
                store.add_obstacle(x, y)
    # endregion book:store-fulfillment-demo-setup

    # region book:store-fulfillment-demo-items
    # Create items
    items = []
    # Grocery items
    for i in range(20):
        x = random.randint(6, 14)
        y = random.randint(6, 14)
        items.append(Item(f"G{i}", f"Grocery Item {i}", "grocery", (x, y)))

    # Produce items
    for i in range(15):
        x = random.randint(21, 29)
        y = random.randint(6, 14)
        items.append(
            Item(
                f"P{i}",
                f"Produce Item {i}",
                "produce",
                (x, y),
                temperature_zone="refrigerated",
                handling_time=1.2,
            )
        )

    # Dairy items
    for i in range(10):
        x = random.randint(36, 44)
        y = random.randint(6, 14)
        items.append(
            Item(
                f"D{i}",
                f"Dairy Item {i}",
                "dairy",
                (x, y),
                temperature_zone="refrigerated",
                handling_time=1.1,
            )
        )
    # endregion book:store-fulfillment-demo-items

    # region book:store-fulfillment-demo-frozen
    # Frozen items
    for i in range(12):
        x = random.randint(6, 14)
        y = random.randint(21, 29)
        items.append(
            Item(
                f"F{i}",
                f"Frozen Item {i}",
                "frozen",
                (x, y),
                temperature_zone="frozen",
                handling_time=1.3,
            )
        )

    # Electronics items
    for i in range(8):
        x = random.randint(21, 29)
        y = random.randint(21, 29)
        items.append(
            Item(
                f"E{i}",
                f"Electronics Item {i}",
                "electronics",
                (x, y),
                handling_time=1.5,
                fragility=0.8,
            )
        )

    # Apparel items
    for i in range(15):
        x = random.randint(36, 44)
        y = random.randint(21, 29)
        items.append(
            Item(
                f"A{i}",
                f"Apparel Item {i}",
                "apparel",
                (x, y),
                handling_time=1.4,
                fragility=0.3,
            )
        )
    # endregion book:store-fulfillment-demo-frozen

    # region book:store-fulfillment-demo-orders
    # Create orders
    orders = []
    for i in range(10):
        # Randomly select 3-8 items for each order
        num_items = random.randint(3, 8)
        order_items = random.sample(items, num_items)
        priority = random.randint(1, 3)
        due_time = random.randint(30, 120)  # Due in 30-120 minutes
        orders.append(Order(f"ORD{i}", order_items, priority, due_time))

    # Create associates
    associates = [
        Associate(
            "A1",
            "Alex",
            efficiency=1.2,
            authorized_zones=["ambient", "refrigerated", "frozen"],
            current_location=(0, 0),
            shift_end_time=240,
        ),
        Associate(
            "A2",
            "Bailey",
            efficiency=1.0,
            authorized_zones=["ambient", "refrigerated"],
            current_location=(0, 20),
            shift_end_time=180,
        ),
        Associate(
            "A3",
            "Casey",
            efficiency=0.9,
            authorized_zones=["ambient"],
            current_location=(25, 0),
            shift_end_time=120,
        ),
    ]

    # endregion book:store-fulfillment-demo-orders

    # region book:store-fulfillment-demo-planner
    # Create fulfillment planner
    planner = FulfillmentPlanner(store)

    for order in orders:
        planner.add_order(order)

    for associate in associates:
        planner.add_associate(associate)

    start_time = time.time()
    planner.plan()
    end_time = time.time()

    print(f"Plan generated in {end_time - start_time:.3f} seconds")
    print(planner.explain_plan())
    planner.visualize_plan()

    return planner
    # endregion book:store-fulfillment-demo-planner


# region book:store-fulfillment-demo-run
# Uncomment to run the demo
# demo_fulfillment_system()
# endregion book:store-fulfillment-demo-run
