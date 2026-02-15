from musciclaim.panels import PanelWhitelist, normalize_panel_label, normalize_panel_list


def test_normalize_panel_label_variants():
    assert normalize_panel_label("A") == "Panel A"
    assert normalize_panel_label("panel a") == "Panel A"
    assert normalize_panel_label("PanelA") == "Panel A"


def test_normalize_panel_list_invalid_flag():
    panels, invalid = normalize_panel_list(["A", "left"])
    assert panels == ["Panel A"]
    assert invalid is True


def test_panel_whitelist_az():
    wl = PanelWhitelist.az()
    assert wl.validate(["Panel A"]) is True
    assert wl.validate(["Panel AA"]) is False
