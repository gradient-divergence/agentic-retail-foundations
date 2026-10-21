import runpy


def test_seven_day_run_waits_at_threshold_and_receives_one_65_unit_order(capsys):
    result = runpy.run_module("demos.intro_inventory_book_demo", run_name="__main__")

    expected = (
        "Day 1: Stock level = 60 "
        "Day 2: Stock level = 55 "
        "Day 3: Stock level = 50 "
        "Day 4: Stock level = 35 Placing order for 65 units. "
        "Day 5: Stock level = 95 "
        "Day 6: Stock level = 90 "
        "Day 7: Stock level = 85"
    ).split()
    assert capsys.readouterr().out.split() == expected
    assert result["agent"].current_stock == 85
    assert result["environment"] == {"stock_level": 80}
