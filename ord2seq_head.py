from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class Ord2SeqOrdinalHead(nn.Module):
    """Pluggable Ord2Seq-style ordinal head.

    Design:
    - Input feature: [B, C]
    - Build a balanced binary hierarchy over class indices [0..K-1]
    - Convert each class label to a sequence of node/group tokens
    - Autoregressive decoder predicts one step at a time
    - Each step outputs class-wise logits [B, K] with BCE targets as group masks
    - Masked decision strategy constrains next-step candidates by previous decision
    """

    def __init__(
        self,
        in_features: int,
        num_classes: int = 4,
        d_model: int = 256,
        nhead: int = 8,
        num_decoder_layers: int = 2,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        use_masked_decision: bool = True,
        step_loss_weights: Optional[List[float]] = None,
    ) -> None:
        super().__init__()
        if num_classes < 2:
            raise ValueError("num_classes must be >= 2 for ordinal classification")

        self.num_classes = int(num_classes)
        self.use_masked_decision = bool(use_masked_decision)

        hierarchy = self._build_hierarchy(self.num_classes)
        self.partitions: List[List[List[int]]] = hierarchy["partitions"]
        self.class_to_group: List[List[int]] = hierarchy["class_to_group"]
        self.num_steps = len(self.partitions)

        if self.num_steps < 1:
            raise RuntimeError("invalid hierarchy: no decoding steps")

        # Token ids: 0 reserved for SOS, all group tokens start from 1.
        token_offsets = []
        cur = 1
        for groups in self.partitions:
            token_offsets.append(cur)
            cur += len(groups)
        self.sos_id = 0
        self.token_vocab_size = cur

        # [K, S]: token id path for each class.
        path_tokens = torch.zeros(self.num_classes, self.num_steps, dtype=torch.long)
        # [K, S, K]: BCE targets per step.
        step_targets = torch.zeros(self.num_classes, self.num_steps, self.num_classes, dtype=torch.float32)
        for c in range(self.num_classes):
            for s in range(self.num_steps):
                gidx = self.class_to_group[s][c]
                path_tokens[c, s] = token_offsets[s] + gidx
                step_targets[c, s, self.partitions[s][gidx]] = 1.0

        self.register_buffer("path_tokens", path_tokens, persistent=True)
        self.register_buffer("step_targets", step_targets, persistent=True)

        if step_loss_weights is None:
            step_loss_weights = [1.0] * self.num_steps
        if len(step_loss_weights) != self.num_steps:
            raise ValueError("step_loss_weights length must equal num_steps")
        self.register_buffer(
            "step_loss_weights",
            torch.tensor(step_loss_weights, dtype=torch.float32),
            persistent=True,
        )

        self.feature_proj = nn.Linear(in_features, d_model)
        self.token_embed = nn.Embedding(self.token_vocab_size, d_model)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="relu",
            batch_first=False,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)
        self.step_classifiers = nn.ModuleList([nn.Linear(d_model, self.num_classes) for _ in range(self.num_steps)])

    @staticmethod
    def _build_hierarchy(num_classes: int) -> Dict[str, List]:
        """Build a balanced binary hierarchy over contiguous class indices."""
        groups = [list(range(num_classes))]
        partitions: List[List[List[int]]] = []
        class_to_group: List[List[int]] = []

        while any(len(g) > 1 for g in groups):
            next_groups: List[List[int]] = []
            c2g = [0] * num_classes

            for g in groups:
                if len(g) <= 1:
                    children = [g]
                else:
                    mid = max(1, len(g) // 2)
                    children = [g[:mid], g[mid:]]

                for child in children:
                    if not child:
                        continue
                    gid = len(next_groups)
                    next_groups.append(child)
                    for c in child:
                        c2g[c] = gid

            partitions.append(next_groups)
            class_to_group.append(c2g)
            groups = next_groups

        return {"partitions": partitions, "class_to_group": class_to_group}

    @staticmethod
    def _causal_mask(size: int, device: torch.device) -> torch.Tensor:
        mask = torch.triu(torch.ones(size, size, device=device), diagonal=1)
        return mask.masked_fill(mask == 1, float("-inf"))

    def _decode_training(self, memory: torch.Tensor, labels: torch.Tensor) -> List[torch.Tensor]:
        """Teacher-forcing decode; returns per-step class logits [B, K]."""
        bsz = labels.shape[0]
        teacher_tokens = self.path_tokens[labels]  # [B, S]
        sos = torch.full((bsz, 1), self.sos_id, dtype=torch.long, device=labels.device)
        if self.num_steps > 1:
            decoder_in = torch.cat([sos, teacher_tokens[:, :-1]], dim=1)
        else:
            decoder_in = sos

        tgt = self.token_embed(decoder_in).transpose(0, 1)  # [S, B, D]
        out = self.decoder(tgt=tgt, memory=memory, tgt_mask=self._causal_mask(self.num_steps, labels.device))

        logits = []
        for s in range(self.num_steps):
            logits.append(self.step_classifiers[s](out[s]))
        return logits

    def _decode_inference(self, memory: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Autoregressive decode with optional masked decision strategy."""
        device = memory.device
        bsz = memory.shape[1]

        tokens = torch.full((bsz, 1), self.sos_id, dtype=torch.long, device=device)
        allowed_mask = torch.ones(bsz, self.num_classes, dtype=torch.bool, device=device)

        step_logits: List[torch.Tensor] = []
        for s in range(self.num_steps):
            tgt = self.token_embed(tokens).transpose(0, 1)
            out = self.decoder(tgt=tgt, memory=memory, tgt_mask=self._causal_mask(tokens.shape[1], device))
            raw_logits = self.step_classifiers[s](out[-1])  # [B, K]

            if self.use_masked_decision:
                logits = raw_logits.masked_fill(~allowed_mask, float("-inf"))
            else:
                logits = raw_logits
            step_logits.append(logits)

            pred_cls = logits.argmax(dim=-1)

            if s < self.num_steps - 1:
                next_token = self.path_tokens[pred_cls, s].unsqueeze(1)
                tokens = torch.cat([tokens, next_token], dim=1)
                if self.use_masked_decision:
                    allowed_mask = self.step_targets[pred_cls, s].bool()

        final_logits = step_logits[-1]
        pred = final_logits.argmax(dim=-1)
        prob = F.softmax(final_logits, dim=-1)
        return {
            "pred": pred,
            "prob": prob,
            "final_logits": final_logits,
            "step_logits": torch.stack(step_logits, dim=1),  # [B, S, K]
        }

    def forward(self, feat: torch.Tensor, labels: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        if feat.dim() != 2:
            raise ValueError("Ord2SeqOrdinalHead expects feature shape [B, C]")

        feat = self.feature_proj(feat)  # [B, D]
        memory = feat.unsqueeze(0)  # [1, B, D]

        out = self._decode_inference(memory)

        if labels is not None:
            labels = labels.long()
            logits_per_step = self._decode_training(memory, labels)
            targets = self.step_targets[labels]  # [B, S, K]

            step_losses = []
            for s in range(self.num_steps):
                loss_s = F.binary_cross_entropy_with_logits(logits_per_step[s], targets[:, s, :], reduction="mean")
                step_losses.append(loss_s)
            step_losses_t = torch.stack(step_losses)
            weights = self.step_loss_weights / self.step_loss_weights.sum().clamp_min(1e-6)
            loss = (step_losses_t * weights).sum()

            out["loss"] = loss
            out["loss_steps"] = step_losses_t

        return out
