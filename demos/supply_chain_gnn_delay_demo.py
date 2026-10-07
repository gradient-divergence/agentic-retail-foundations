"""
Minimal GNN demo for delay propagation on a supply chain graph.
"""

from __future__ import annotations

try:
    import torch
    from torch_geometric.data import Data
    from torch_geometric.nn import GCNConv
except ModuleNotFoundError:
    raise SystemExit("Install the gnn extra: uv sync --extra gnn") from None


class DelayPropagationGNN(torch.nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, data: Data) -> torch.Tensor:
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)
        return x


def build_demo_graph() -> Data:
    # Node features: [lead_time_days, capacity_utilization]
    x = torch.tensor(
        [
            [2.0, 0.7],  # supplier
            [1.0, 0.8],  # port
            [3.0, 0.6],  # DC
            [1.5, 0.9],  # store
        ],
        dtype=torch.float,
    )

    # Directed edges: supplier -> port -> DC -> store
    edge_index = torch.tensor(
        [
            [0, 1, 2],
            [1, 2, 3],
        ],
        dtype=torch.long,
    )

    return Data(x=x, edge_index=edge_index)


def run_demo() -> None:
    data = build_demo_graph()
    model = DelayPropagationGNN(in_channels=2, hidden_channels=4, out_channels=1)
    predicted_delay = model(data).squeeze(-1)
    print("Predicted delay signal per node:", predicted_delay.tolist())


if __name__ == "__main__":
    run_demo()
