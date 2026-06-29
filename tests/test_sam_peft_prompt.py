from __future__ import annotations

import torch
import torch.nn as nn

from r6.models.prompt_generator import PromptGenerator
from r6.models.real_sam_wrapper import RealSAMWrapper
from r6.models.sam_peft import BlockWithAdapter, LoRALinear, SAMPEFTAdapter


class TinyBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = nn.Module()
        self.attn.qkv = nn.Linear(8, 24)
        self.attn.proj = nn.Linear(8, 8)


class TinySAM(nn.Module):
    def __init__(self):
        super().__init__()
        self.image_encoder = nn.Module()
        self.image_encoder.blocks = nn.ModuleList([TinyBlock(), TinyBlock()])
        self.mask_decoder = nn.Linear(8, 8)


def test_lora_injection_targets_last_blocks_only():
    sam = TinySAM()
    adapter = SAMPEFTAdapter(
        sam,
        train_peft=True,
        peft_type="lora",
        train_mask_decoder=False,
        train_last_n_blocks=1,
        lora_rank=2,
        lora_alpha=4,
        max_trainable_ratio=1.0,
        hard_max_trainable_ratio=1.0,
    )
    assert not isinstance(sam.image_encoder.blocks[0].attn.qkv, LoRALinear)
    assert isinstance(sam.image_encoder.blocks[1].attn.qkv, LoRALinear)
    assert isinstance(sam.image_encoder.blocks[1].attn.proj, LoRALinear)
    assert adapter.report.lora_param_count > 0


def test_adapter_injection_targets_last_blocks_only():
    sam = TinySAM()
    adapter = SAMPEFTAdapter(
        sam,
        train_peft=True,
        peft_type="adapter",
        train_mask_decoder=False,
        train_last_n_blocks=1,
        adapter_dim=3,
        max_trainable_ratio=1.0,
        hard_max_trainable_ratio=1.0,
    )
    assert not isinstance(sam.image_encoder.blocks[0], BlockWithAdapter)
    assert isinstance(sam.image_encoder.blocks[1], BlockWithAdapter)
    assert adapter.report.adapter_param_count > 0
    assert adapter.report.lora_param_count == 0


class FakePELayer(nn.Module):
    def forward_with_coords(self, coords, image_size):
        out = torch.zeros(*coords.shape[:-1], 256, device=coords.device)
        out[..., :2] = coords
        return out


class FakePromptEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_image_size = (1024, 1024)
        self.pe_layer = FakePELayer()
        self.point_embeddings = nn.ModuleList([nn.Embedding(1, 256) for _ in range(4)])


def test_local_prompt_preparation_uses_points_negative_points_and_boxes():
    wrapper = RealSAMWrapper.__new__(RealSAMWrapper)
    nn.Module.__init__(wrapper)
    wrapper.device = torch.device("cpu")
    wrapper.image_size = 1024
    wrapper.sam_source = "local Model/sam:test"
    wrapper.use_point_prompt = True
    wrapper.use_negative_points = True
    wrapper.use_box_prompt = True
    wrapper.sam = nn.Module()
    wrapper.sam.prompt_encoder = FakePromptEncoder()
    prompts = {
        "point_coords": torch.tensor([[[0.25, 0.50]]]),
        "point_labels": torch.ones(1, 1, dtype=torch.long),
        "negative_point_coords": torch.tensor([[[0.75, 0.10]]]),
        "boxes_xyxy": torch.tensor([[0.1, 0.2, 0.8, 0.9]]),
    }
    points, boxes = wrapper._prepare_sparse_prompts(prompts)
    assert points[0].shape == (1, 2, 256)
    assert points[1].tolist() == [[1, 0]]
    assert boxes.shape == (1, 2, 256)


def test_prompt_generator_invalid_foreground_uses_compact_fallback_box():
    generator = PromptGenerator(
        num_classes=3,
        mask_prompt_size=8,
        min_component_area=4,
        residual_scale=0.0,
        box_threshold=0.80,
        fallback_box_half_size=0.05,
    )
    image = torch.zeros(1, 3, 8, 8)
    teacher_prob = torch.zeros(1, 3, 8, 8)
    teacher_prob[:, 0] = 0.98
    teacher_prob[:, 1] = 0.01
    teacher_prob[:, 2] = 0.01

    out = generator(image=image, teacher_prob=teacher_prob, mode="unlabeled")

    boxes = out["boxes_xyxy"]
    box_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    assert torch.all(box_area < 0.02)
    assert torch.all(out["prompt_valid"][:, 1:] == 0)
    assert torch.all(out["prompt_quality"][:, 1:] == 0)
