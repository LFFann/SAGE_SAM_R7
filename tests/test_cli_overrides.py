from __future__ import annotations

import pytest

from train_r6 import apply_cli_overrides


def test_apply_cli_overrides_parses_yaml_scalars_and_lists():
    config = {"sam": {"use_sam": True}, "losses": {"x": {"enabled": True}}, "values": {}}

    apply_cli_overrides(
        config,
        [
            "sam.use_sam",
            "false",
            "losses.x.weight",
            "0.25",
            "values.classes",
            "[1, 2]",
            "values.name",
            "demo",
        ],
    )

    assert config["sam"]["use_sam"] is False
    assert config["losses"]["x"]["weight"] == 0.25
    assert config["values"]["classes"] == [1, 2]
    assert config["values"]["name"] == "demo"


def test_apply_cli_overrides_rejects_odd_pairs():
    with pytest.raises(ValueError, match="KEY VALUE"):
        apply_cli_overrides({}, ["sam.use_sam"])
