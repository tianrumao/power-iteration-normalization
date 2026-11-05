import torch
from torch import nn


def unit(vector, dim):
    length = torch.linalg.vector_norm(vector, dim=dim, keepdim=True)
    valid = torch.isfinite(vector).all(dim=dim, keepdim=True)
    valid = valid & torch.isfinite(length) & (length > 0)
    numerator = torch.where(valid, vector, 0.0)
    denominator = torch.where(valid, length, 1.0)
    return numerator / denominator, valid.squeeze(dim)


def normalize_tokens(tokens):
    normalized, valid = unit(tokens, -2)
    return torch.where(valid.unsqueeze(-2), normalized, tokens)


def match_lengths(tokens, reference):
    directions, valid = unit(tokens, -2)
    wanted = torch.linalg.vector_norm(reference, dim=-2)
    finite = torch.isfinite(tokens).all(dim=-2) & torch.isfinite(wanted)
    zero = (tokens == 0).all(dim=-2)
    possible = finite & (valid | (zero & (wanted == 0)))
    return directions * wanted.unsqueeze(-2), possible.all(dim=-1)


def readout(tokens):
    d = tokens.shape[-2] // 2
    return unit(tokens[:, d:, -1], -1)


def direction_error(prediction, target, valid):
    alignment = (prediction * target).sum(dim=-1).clamp(-1, 1)
    return torch.where(valid, 1 - alignment.square(), 1.0)


class Block(nn.Module):
    def __init__(self, width, std):
        super().__init__()
        self.v = nn.Parameter(torch.randn(width, width) * std)
        self.w = nn.Parameter(torch.randn(width, width) * std)

    def forward(self, tokens):
        return tokens + self.v @ ((tokens @ tokens.transpose(-1, -2)) @ (self.w @ tokens))


class Model(nn.Module):
    def __init__(self, d, norm_on, init_std):
        super().__init__()
        self.blocks = nn.ModuleList(Block(2 * d, init_std) for _ in range(4))
        self.norm_on = norm_on

    def forward(self, tokens, bypass=None, reference=None):
        states = []
        possible = torch.ones(tokens.shape[0], dtype=torch.bool, device=tokens.device)
        for layer, block in enumerate(self.blocks):
            if layer != bypass:
                tokens = block(tokens)
            if self.norm_on and layer < 3:
                tokens = normalize_tokens(tokens)
            if layer == bypass and reference is not None:
                tokens, possible = match_lengths(tokens, reference)
            states.append(tokens)
        return states, possible
