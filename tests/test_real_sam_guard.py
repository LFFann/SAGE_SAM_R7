from __future__ import annotations

from pathlib import Path

import pytest
import torch

import r6.models.real_sam_wrapper as real_sam_wrapper
from r6.models.real_sam_wrapper import RealSAMWrapper
from r6.ssl.experimental_sparse_sam_relation_graph import build_topk_relation_graph


def test_local_sam_root_is_detected_from_installed_package_path():
    root = Path(real_sam_wrapper.__file__).resolve().parents[2]

    assert real_sam_wrapper._find_local_r6_root() == root


def test_missing_sam_checkpoint_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        RealSAMWrapper("vit_b", tmp_path / "missing.pth", "cpu")


def test_dense_relation_graph_forbidden():
    emb = torch.randn(1, 4, 2, 2)
    with pytest.raises(ValueError):
        build_topk_relation_graph(emb, topk=4)
