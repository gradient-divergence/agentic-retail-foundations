from demos import structured_output_schema_demo as demo


def test_malformed_proposal_routes_to_human(monkeypatch, capsys):
    monkeypatch.setattr(demo, "propose_price_update", lambda: {"sku": "SKU-1001", "new_price": -1})
    demo.run_pipeline()
    output = capsys.readouterr().out
    assert "route_to_human_review" in output
    assert "Apply price" not in output


def test_proposal_validation_error_routes_to_human(monkeypatch, capsys):
    def invalid_proposal():
        return demo.PriceUpdatePayload(sku="SKU-1001", new_price=-1, reason="bad", confidence=0.5)

    monkeypatch.setattr(demo, "propose_price_update", invalid_proposal)
    demo.run_pipeline()
    assert "route_to_human_review" in capsys.readouterr().out


def test_extra_fields_do_not_disappear(monkeypatch, capsys):
    def invalid_proposal():
        return demo.PriceUpdatePayload(
            sku="SKU-1001", new_price=1, reason="bad", confidence=0.5, unapproved=True
        )

    monkeypatch.setattr(demo, "propose_price_update", invalid_proposal)
    demo.run_pipeline()
    assert "route_to_human_review" in capsys.readouterr().out


def test_valid_pipeline_prints_original_result(capsys):
    demo.run_pipeline()
    assert capsys.readouterr().out == "Apply price: SKU-1001 -> 109.0\n"
