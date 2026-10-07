from demos.supply_chain_gnn_delay_demo import DelayPropagationGNN, build_demo_graph


def test_demo_graph_has_directed_supply_chain_and_finite_node_outputs():
    data = build_demo_graph()
    assert data.x.shape == (4, 2)
    assert data.edge_index.tolist() == [[0, 1, 2], [1, 2, 3]]
    output = DelayPropagationGNN(2, 4, 1)(data)
    assert output.shape == (4, 1)
    assert output.isfinite().all()
