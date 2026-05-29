import torch

from .GeoTokenV2CAGHead import GeoTokenV2CAGHead


class GeoTokenV2CAGDivHead(GeoTokenV2CAGHead):
    def _make_tokens_with_attn(self, x):
        batch = x.shape[0]
        global_token = x.mean(dim=(2, 3)).unsqueeze(1)

        feat = x.flatten(2).transpose(1, 2)
        query = self.query.unsqueeze(0).expand(batch, -1, -1)
        q = self.q_proj(query)
        k = self.k_proj(feat)
        v = self.v_proj(feat)
        attn = torch.softmax(torch.matmul(q, k.transpose(1, 2)) * self.scale, dim=-1)
        query_tokens = torch.matmul(attn, v)

        detail_map = self.detail_dwconv(x)
        detail_token = detail_map.mean(dim=(2, 3)).unsqueeze(1)
        tokens = torch.cat([global_token, query_tokens, detail_token], dim=1)
        return tokens, attn

    def forward(self, x):
        if x.ndim != 4:
            raise ValueError(
                "GeoTokenV2CAGDivHead expects [B, C, H, W] feature map, got {}".format(
                    tuple(x.shape)
                )
            )

        batch = x.shape[0]
        tokens, attn = self._make_tokens_with_attn(x)
        gate = torch.sigmoid(self.gate_mlp(tokens))
        gated_tokens = tokens * gate

        embedding_raw = self.embedding(gated_tokens.reshape(batch, -1))
        embedding = self.bnneck(embedding_raw)
        cls = self.classifier(embedding)
        aux = {"attn": attn}
        return [cls, embedding, aux]
