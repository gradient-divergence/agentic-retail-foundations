"""
Inventory collaboration network protocol agent/logic.
Coordinates inventory transfers between stores.
"""

from datetime import datetime
import random
from typing import TypedDict

from models.store import Store


class InventoryCollaborationNetwork:
    """
    Manages a network of stores and coordinates inventory transfers based on needs, excess, and transfer costs.
    """

    def __init__(self, max_transfer_distance: float = 100.0):
        """
        Args:
            max_transfer_distance: Maximum allowed transfer cost/distance between stores.
        """
        self.stores: dict[str, Store] = {}
        self.max_transfer_distance = max_transfer_distance
        self.transfer_costs: dict[tuple[str, str], float] = {}
        self.pending_transfers: list[dict] = []

    def register_store(self, store: Store) -> None:
        """
        Register a store in the network and calculate transfer costs to/from all other stores.
        Args:
            store: Store instance to register.
        """
        self.stores[store.store_id] = store
        for eid, estore in self.stores.items():
            if eid != store.store_id:
                cost = (
                    store.transfer_cost_factor
                    * estore.transfer_cost_factor
                    * random.uniform(0.5, 2.0)
                )
                self.transfer_costs[(store.store_id, eid)] = cost
                self.transfer_costs[(eid, store.store_id)] = cost

    async def identify_transfer_opportunities(self) -> list[dict]:
        """
        Identify inventory transfer opportunities between stores based on needs, excess, and transfer costs.
        Returns:
            List of proposed transfer operations (dicts).
        """
        opportunities = []
        store_needs: dict[str, dict[str, int]] = {}
        store_excess: dict[str, dict[str, int]] = {}

        # Define structure for clarity and type checking
        class PotentialSenderInfo(TypedDict):
            sender_id: str
            available_qty: int
            transfer_cost: float
            net_value: float
            value_per_unit: float

        for sid, st in self.stores.items():
            store_needs[sid] = st.get_needed_inventory()
            store_excess[sid] = st.get_sharable_inventory()

        for needing_id, needs in store_needs.items():
            needing_store = self.stores[needing_id]
            for product_id, qty_needed in needs.items():
                potential_senders: list[PotentialSenderInfo] = []
                for sending_id, excess in store_excess.items():
                    if sending_id == needing_id:
                        continue
                    if product_id in excess and excess[product_id] > 0:
                        sending_store = self.stores[sending_id]
                        transfer_cost = self.transfer_costs.get(
                            (sending_id, needing_id), float("inf")
                        )
                        if transfer_cost > self.max_transfer_distance:
                            continue
                        available = min(excess[product_id], qty_needed)
                        sender_val = sending_store.calculate_transfer_value(
                            product_id, available, True
                        )
                        receiver_val = needing_store.calculate_transfer_value(
                            product_id, available, False
                        )
                        net_val = (
                            sender_val + receiver_val - (transfer_cost * available)
                        )
                        if net_val > 0 and available > 0:
                            sender_info: PotentialSenderInfo = {
                                "sender_id": sending_id,
                                "available_qty": available,
                                "transfer_cost": transfer_cost,
                                "net_value": net_val,
                                "value_per_unit": net_val / available,
                            }
                            potential_senders.append(sender_info)
                potential_senders.sort(key=lambda x: x["value_per_unit"], reverse=True)
                rem_need: int = qty_needed
                for ps in potential_senders:
                    if rem_need <= 0:
                        break
                    tr_qty: int = min(ps["available_qty"], rem_need)
                    opportunities.append(
                        {
                            "sender_id": ps["sender_id"],
                            "receiver_id": needing_id,
                            "product_id": product_id,
                            "quantity": tr_qty,
                            "transfer_cost": ps["transfer_cost"] * tr_qty,
                            "net_value": ps["value_per_unit"] * tr_qty,
                            "status": "proposed",
                        }
                    )
                    rem_need -= tr_qty
                    sender_id_key: str = ps["sender_id"]
                    store_excess[sender_id_key][product_id] -= tr_qty
        return opportunities

    async def execute_transfers(self, approved_ops: list[dict]) -> list[dict]:
        """
        Execute approved inventory transfers between stores.
        Args:
            approved_ops: List of approved transfer operations (dicts).
        Returns:
            List of results for each transfer (dicts with status and timestamp).
        """
        results = []
        for op in approved_ops:
            sid = op["sender_id"]
            rid = op["receiver_id"]
            pid = op["product_id"]
            qty = op["quantity"]
            s = self.stores[sid]
            r = self.stores[rid]
            send_ok = s.execute_transfer(pid, qty, rid, True)
            recv_ok = r.execute_transfer(pid, qty, sid, False)
            success = send_ok and recv_ok
            op_res = op.copy()
            op_res["status"] = "completed" if success else "failed"
            op_res["timestamp"] = datetime.now()
            results.append(op_res)
        return results
