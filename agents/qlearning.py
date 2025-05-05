"""
Q-learning agent for dynamic pricing MDP.
Refactored for modular use in agentic-retail-foundations.
"""

from collections import defaultdict
import numpy as np
import pickle
import pandas as pd
from config.config import QLearningAgentConfig
from utils.logger import get_logger


class QLearningAgent:
    """
    A Q-learning agent for solving the Dynamic Pricing MDP.

    Q-learning is a model-free reinforcement learning algorithm that learns
    a policy by directly estimating the Q-values (expected future rewards)
    for each state-action pair.
    """

    def __init__(self, config: QLearningAgentConfig):
        self.config = config
        self.logger = get_logger(self.__class__.__name__)
        self.q_table: dict[tuple[int, int, int], np.ndarray] = defaultdict(
            lambda: np.zeros(self.config.action_space_size)
        )
        self.learning_rate = config.learning_rate
        self.discount_factor = config.discount_factor
        self.exploration_rate = config.exploration_rate
        self.min_exploration_rate = config.min_exploration_rate
        self.exploration_decay = config.exploration_decay
        self.action_space_size = config.action_space_size
        self.logger.info(
            f"QLearningAgent initialized: LR={self.learning_rate}, Gamma={self.discount_factor}, Epsilon={self.exploration_rate}"
        )

    def choose_action(
        self, state: tuple, available_actions: list[int] | None = None
    ) -> int:
        if available_actions is None:
            available_actions = list(range(self.action_space_size))
        if not available_actions:
            self.logger.error("No available actions to choose from.")
            raise ValueError("No available actions.")
        if np.random.random() < self.exploration_rate:
            action = int(np.random.choice(available_actions))
            self.logger.debug(f"Action chosen (Explore): {action} from state {state}")
            return action
        state_q_values = self.q_table[state]
        available_q_values = {
            action: state_q_values[action] for action in available_actions
        }
        max_q = -np.inf
        if available_q_values:
            max_q = max(available_q_values.values())
        best_actions = [
            action for action, q in available_q_values.items() if q == max_q
        ]
        action_choice = (
            np.random.choice(best_actions)
            if best_actions
            else np.random.choice(available_actions)
        )
        action = int(action_choice)
        self.logger.debug(
            f"Action chosen (Exploit): {action} (Q={max_q:.3f}) from state {state}"
        )
        return action  # type: ignore[no-any-return]

    def update(
        self, state: tuple, action: int, reward: float, next_state: tuple, done: bool
    ):
        next_state_q_values = self.q_table[next_state]
        max_next_q = np.max(next_state_q_values) if not done else 0.0
        td_target = reward + self.discount_factor * max_next_q
        td_error = td_target - self.q_table[state][action]
        self.q_table[state][action] += self.learning_rate * td_error
        self.logger.debug(
            f"Q-update: State={state}, Action={action}, Reward={reward:.2f}, NextState={next_state}, Done={done}, TD_Error={td_error:.3f}, New Q={self.q_table[state][action]:.3f}"
        )

    def decay_exploration(self):
        if self.exploration_rate > self.min_exploration_rate:
            self.exploration_rate *= self.exploration_decay
            self.exploration_rate = max(
                self.min_exploration_rate, self.exploration_rate
            )

    def get_policy(self) -> dict[tuple, int]:
        policy = {}
        for state, q_values in self.q_table.items():
            best_action_idx: int = int(np.argmax(q_values))
            policy[state] = best_action_idx  # type: ignore[assignment]
        return policy

    def get_q_table_df(self, discount_map: list[float]) -> pd.DataFrame | None:
        if not self.q_table:
            return None
        records = []
        for state, q_values in self.q_table.items():
            weeks_rem, inventory, current_disc_idx = state
            record = {
                "weeks_remaining": weeks_rem,
                "inventory_level": inventory,
                "current_discount_idx": current_disc_idx,
            }
            for action_idx, q_val in enumerate(q_values):
                record[f"Q(disc={discount_map[action_idx] * 100:.0f}%)"] = q_val
            record["best_action_idx"] = int(np.argmax(q_values))
            record["best_discount"] = int(discount_map[record["best_action_idx"]] * 100)
            records.append(record)
        df = pd.DataFrame(records)
        df = df.sort_values(
            by=["weeks_remaining", "inventory_level", "current_discount_idx"],
            ascending=[False, True, True],
        )
        return df.reset_index(drop=True)

    def save(self, filepath: str):
        """Serialize the agent state and Q-table to a file."""
        with open(filepath, "wb") as f:
            pickle.dump(
                {
                    "q_table": dict(self.q_table),
                    "learning_rate": self.learning_rate,
                    "discount_factor": self.discount_factor,
                    "exploration_rate": self.exploration_rate,
                    "min_exploration_rate": self.min_exploration_rate,
                    "exploration_decay": self.exploration_decay,
                    "action_space_size": self.action_space_size,
                },
                f,
            )
        self.logger.info(f"Agent state saved to {filepath}")

    def load(self, filepath: str):
        """Load the agent state and Q-table from a file."""
        with open(filepath, "rb") as f:
            state = pickle.load(f)
            self.q_table = defaultdict(
                lambda: np.zeros(state["action_space_size"]), state["q_table"]
            )
            self.learning_rate = state["learning_rate"]
            self.discount_factor = state["discount_factor"]
            self.exploration_rate = state["exploration_rate"]
            self.min_exploration_rate = state["min_exploration_rate"]
            self.exploration_decay = state["exploration_decay"]
            self.action_space_size = state["action_space_size"]
        self.logger.info(f"Agent state loaded from {filepath}")
