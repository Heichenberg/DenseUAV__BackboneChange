import math

import torch
import torch.nn as nn

from .utils import weights_init_kaiming, weights_init_classifier


class GeoTokenHeadV1(nn.Module):
    def __init__(self, opt):
        super().__init__()
        in_channels = opt.in_planes
        self.proj_dim = getattr(opt, "geo_proj_dim", 384)
        self.embedding_dim = opt.num_bottleneck
        self.center_size = getattr(opt, "geo_center_size", 3)
        self.context_size = getattr(opt, "geo_context_size", 7)
        self.context_dim = getattr(opt, "geo_context_dim", self.proj_dim)
        self.context_gate_enabled = getattr(opt, "geo_context_gate", False)
        self.context_gate_init = getattr(opt, "geo_context_gate_init", 0.1)
        self.drop_rate = getattr(opt, "geo_drop_rate", 0.0)
        active_tokens = getattr(
            opt,
            "geo_active_tokens",
            ("global", "center", "context", "structure"),
        )
        self.active_tokens = tuple(active_tokens)
        valid_tokens = {"global", "center", "context", "structure"}
        if not self.active_tokens:
            raise ValueError("GeoTokenHeadV1 requires at least one active token")
        invalid_tokens = [name for name in self.active_tokens if name not in valid_tokens]
        if invalid_tokens:
            raise ValueError(
                "Unsupported GeoTokenHeadV1 tokens: {}. Valid: {}".format(
                    invalid_tokens, sorted(valid_tokens)
                )
            )
        if self.context_size not in (5, 7):
            raise ValueError("GeoTokenHeadV1 geo_context_size currently supports only 5 or 7")

        self.input_norm = nn.LayerNorm(in_channels)
        self.channel_proj = nn.Linear(in_channels, self.proj_dim)
        self.context_proj = (
            nn.Identity()
            if self.context_dim == self.proj_dim
            else nn.Linear(self.proj_dim, self.context_dim)
        )
        if self.context_gate_enabled:
            gate_init = min(max(self.context_gate_init, 1e-4), 1 - 1e-4)
            gate_logit = math.log(gate_init / (1 - gate_init))
            self.context_gate_logit = nn.Parameter(torch.tensor(gate_logit, dtype=torch.float32))
        else:
            self.context_gate_logit = None
        self.structure_score = nn.Conv2d(self.proj_dim, 1, kernel_size=1, bias=True)
        self.local_avg = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)
        token_dims = {
            "global": self.proj_dim,
            "center": self.proj_dim,
            "context": self.context_dim,
            "structure": self.proj_dim,
        }
        self.concat_dim = sum(token_dims[name] for name in self.active_tokens)
        concat_dim = self.concat_dim
        self.embedding = nn.Linear(concat_dim, self.embedding_dim)
        self.dropout = nn.Dropout(self.drop_rate) if self.drop_rate > 0 else nn.Identity()
        self.bnneck = nn.BatchNorm1d(self.embedding_dim)
        self.bnneck.bias.requires_grad_(False)
        self.classifier = nn.Linear(self.embedding_dim, opt.nclasses)
        self.last_x_proj = None
        self.last_center_mask = None
        self.last_context_mask = None
        self.last_structure_attn = None
        self.last_context_gate = None

        self.channel_proj.apply(weights_init_kaiming)
        self.context_proj.apply(weights_init_kaiming)
        self.structure_score.apply(weights_init_kaiming)
        self.embedding.apply(weights_init_kaiming)
        self.bnneck.apply(weights_init_kaiming)
        self.classifier.apply(weights_init_classifier)

    def _project(self, x):
        batch, channels, height, width = x.shape
        tokens = x.flatten(2).transpose(1, 2)
        tokens = self.input_norm(tokens)
        tokens = self.channel_proj(tokens)
        return tokens.transpose(1, 2).reshape(batch, self.proj_dim, height, width)

    def _center_bounds(self, height, width):
        size = min(self.center_size, height, width)
        size = max(1, size)
        row_start = max(0, (height - size) // 2)
        col_start = max(0, (width - size) // 2)
        row_end = min(height, row_start + size)
        col_end = min(width, col_start + size)
        return row_start, row_end, col_start, col_end

    def _region_mask(self, batch, height, width, size, x):
        row_start, row_end, col_start, col_end = self._center_bounds_for_size(height, width, size)
        mask = x.new_zeros((batch, 1, height, width))
        mask[:, :, row_start:row_end, col_start:col_end] = 1
        return mask

    def _center_bounds_for_size(self, height, width, size):
        size = min(size, height, width)
        size = max(1, size)
        row_start = max(0, (height - size) // 2)
        col_start = max(0, (width - size) // 2)
        row_end = min(height, row_start + size)
        col_end = min(width, col_start + size)
        return row_start, row_end, col_start, col_end

    def _masked_average(self, x, mask):
        weighted = x * mask
        denom = mask.sum(dim=(2, 3)).clamp(min=1.0)
        return weighted.sum(dim=(2, 3)) / denom

    def get_context_gate_value(self):
        if self.context_gate_logit is None:
            return None
        return torch.sigmoid(self.context_gate_logit).detach().item()

    def forward(self, x):
        if x.ndim != 4:
            raise ValueError("GeoTokenHeadV1 expects [B, C, H, W], got {}".format(tuple(x.shape)))

        x_proj = self._project(x)
        batch, channels, height, width = x_proj.shape

        global_token = x_proj.mean(dim=(2, 3))

        row_start, row_end, col_start, col_end = self._center_bounds(height, width)
        center_region = x_proj[:, :, row_start:row_end, col_start:col_end]
        center_token = center_region.mean(dim=(2, 3))

        center_mask = self._region_mask(batch, height, width, self.center_size, x_proj)
        outer_mask = self._region_mask(batch, height, width, self.context_size, x_proj)
        context_mask = (outer_mask - center_mask).clamp(min=0)
        if int(context_mask.sum().item()) == 0:
            context_token_raw = global_token
        else:
            context_token_raw = self._masked_average(x_proj, context_mask)
        context_token = self.context_proj(context_token_raw)
        if self.context_gate_logit is not None and "context" in self.active_tokens:
            context_token = torch.sigmoid(self.context_gate_logit) * context_token

        local_avg = self.local_avg(x_proj)
        local_residual = (x_proj - local_avg).abs()
        structure_score = self.structure_score(local_residual).flatten(2)
        structure_attn = torch.softmax(structure_score, dim=-1).view(batch, 1, height, width)
        structure_token = (x_proj * structure_attn).sum(dim=(2, 3))
        self.last_x_proj = x_proj.detach()
        self.last_center_mask = center_mask.detach()
        self.last_context_mask = context_mask.detach()
        self.last_structure_attn = structure_attn.detach()
        if self.context_gate_logit is not None:
            self.last_context_gate = torch.sigmoid(self.context_gate_logit).detach()
        else:
            self.last_context_gate = None

        token_dict = {
            "global": global_token,
            "center": center_token,
            "context": context_token,
            "structure": structure_token,
        }
        selected_tokens = [token_dict[name] for name in self.active_tokens]
        concat = torch.cat(selected_tokens, dim=1)
        embedding = self.embedding(concat)
        embedding = self.dropout(embedding)
        embedding = self.bnneck(embedding)
        cls = self.classifier(embedding)
        return [cls, embedding]


class MGTRF(nn.Module):
    def __init__(self, opt):
        super().__init__()
        in_channels = opt.in_planes
        self.num_queries = getattr(opt, "mgtrf_num_queries", 8)
        self.proj_dim = getattr(opt, "mgtrf_proj_dim", 384)
        self.embedding_dim = opt.num_bottleneck
        self.drop_rate = getattr(opt, "mgtrf_drop_rate", getattr(opt, "droprate", 0.0))

        self.input_norm = nn.LayerNorm(in_channels)
        self.channel_proj = nn.Linear(in_channels, self.proj_dim)
        self.query_tokens = nn.Parameter(torch.randn(self.num_queries, self.proj_dim) * 0.02)
        self.query_norm = nn.LayerNorm(self.proj_dim)
        self.key_proj = nn.Linear(self.proj_dim, self.proj_dim)
        self.value_proj = nn.Linear(self.proj_dim, self.proj_dim)
        self.local_avg = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)

        self.num_tokens = self.num_queries + 2
        self.token_norm = nn.LayerNorm(self.proj_dim)
        self.cag = nn.Sequential(
            nn.Linear(self.proj_dim, max(self.proj_dim // 4, 32)),
            nn.GELU(),
            nn.Linear(max(self.proj_dim // 4, 32), 1),
            nn.Sigmoid(),
        )

        self.embedding = nn.Linear(self.num_tokens * self.proj_dim, self.embedding_dim)
        self.dropout = nn.Dropout(self.drop_rate) if self.drop_rate > 0 else nn.Identity()
        self.bnneck = nn.BatchNorm1d(self.embedding_dim)
        self.bnneck.bias.requires_grad_(False)
        self.classifier = nn.Linear(self.embedding_dim, opt.nclasses)
        self.last_query_attn = None
        self.last_token_gates = None

        self.channel_proj.apply(weights_init_kaiming)
        self.key_proj.apply(weights_init_kaiming)
        self.value_proj.apply(weights_init_kaiming)
        self.cag.apply(weights_init_kaiming)
        self.embedding.apply(weights_init_kaiming)
        self.bnneck.apply(weights_init_kaiming)
        self.classifier.apply(weights_init_classifier)

    def _project(self, x):
        batch, channels, height, width = x.shape
        tokens = x.flatten(2).transpose(1, 2)
        tokens = self.input_norm(tokens)
        tokens = self.channel_proj(tokens)
        return tokens, height, width

    def forward(self, x):
        if x.ndim != 4:
            raise ValueError("MGTRF expects [B, C, H, W], got {}".format(tuple(x.shape)))

        tokens, height, width = self._project(x)
        batch = tokens.shape[0]

        global_token = tokens.mean(dim=1, keepdim=True)

        query = self.query_norm(self.query_tokens).unsqueeze(0).expand(batch, -1, -1)
        key = self.key_proj(tokens)
        value = self.value_proj(tokens)
        attn = torch.softmax(torch.matmul(query, key.transpose(1, 2)) / math.sqrt(self.proj_dim), dim=-1)
        query_tokens = torch.matmul(attn, value)

        feature_map = tokens.transpose(1, 2).reshape(batch, self.proj_dim, height, width)
        detail_map = (feature_map - self.local_avg(feature_map)).abs()
        detail_token = detail_map.flatten(2).mean(dim=2).unsqueeze(1)

        multi_tokens = torch.cat([global_token, query_tokens, detail_token], dim=1)
        gates = self.cag(self.token_norm(multi_tokens))
        multi_tokens = multi_tokens * gates

        embedding = self.embedding(multi_tokens.flatten(1))
        embedding = self.dropout(embedding)
        embedding = self.bnneck(embedding)
        cls = self.classifier(embedding)

        self.last_query_attn = attn.reshape(batch, self.num_queries, height, width).detach()
        self.last_token_gates = gates.detach()
        return [cls, embedding]
