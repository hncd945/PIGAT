#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PhysGAT neural network model architecture with multi-hop graph attention layers.
Implements heterogeneous graph processing for source-receptor pollution attribution.

PhysGAT神经网络模型架构，包含多跳图注意力层。
实现用于污染源-受体归因的异构图处理。

Author: Wenhao Wang
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv

class PhysGATModel(nn.Module):
    def __init__(self, receptor_in_channels: int, source_in_channels: int,
                 hidden_channels: int, out_channels: int, num_heads: int,
                 dropout_rate: float, hops: list[int], **kwargs):
        super().__init__()
        self.hops = sorted(hops)
        self.dropout = nn.Dropout(p=dropout_rate)
        edge_dim = 7

        self.receptor_paths = nn.ModuleDict()
        total_receptor_out_channels = 0
        for hop in self.hops:
            layers = nn.ModuleList()
            in_c = receptor_in_channels
            for i in range(hop):
                out_c = hidden_channels if i < hop - 1 else out_channels
                heads = num_heads if i < hop - 1 else 1
                layers.append(GATv2Conv(in_c, out_c, heads=heads, edge_dim=edge_dim, add_self_loops=False))
                in_c = out_c * heads
            self.receptor_paths[f'hop_{hop}'] = layers
            total_receptor_out_channels += out_c

        self.source_paths = nn.ModuleDict()
        total_source_out_channels = 0
        for hop in self.hops:
            layers = nn.ModuleList()
            in_c = source_in_channels
            for i in range(hop):
                out_c = hidden_channels if i < hop - 1 else out_channels
                heads = num_heads if i < hop - 1 else 1
                layers.append(GATv2Conv(in_c, out_c, heads=heads, add_self_loops=False))
                in_c = out_c * heads
            self.source_paths[f'hop_{hop}'] = layers
            total_source_out_channels += out_c

        self.receptor_norm = nn.BatchNorm1d(total_receptor_out_channels)
        self.source_norm = nn.BatchNorm1d(total_source_out_channels)

        self.cross_conv_s2r = GATv2Conv((total_source_out_channels, total_receptor_out_channels),
                                      total_receptor_out_channels, heads=1, edge_dim=edge_dim, add_self_loops=False)

        self.receptor_out_linear = nn.Linear(total_receptor_out_channels * 2, out_channels)
        self.source_out_linear = nn.Linear(total_source_out_channels, out_channels)

        self.final_receptor_norm = nn.BatchNorm1d(out_channels)
        self.final_source_norm = nn.BatchNorm1d(out_channels)


    def forward(self, x_receptor, x_source, edge_index_rr, edge_index_rs, edge_attr_rr=None, edge_attr_rs=None, return_attention_weights=False):
        receptor_hop_outputs = []
        for hop in self.hops:
            h_r = x_receptor
            path = self.receptor_paths[f'hop_{hop}']
            for i, conv in enumerate(path):
                h_r = conv(h_r, edge_index_rr, edge_attr=edge_attr_rr)
                if i < len(path) - 1:
                    h_r = F.leaky_relu(h_r)
                    h_r = self.dropout(h_r)
            receptor_hop_outputs.append(h_r)

        h_receptor = torch.cat(receptor_hop_outputs, dim=-1)
        h_receptor = self.receptor_norm(h_receptor)

        source_hop_outputs = []
        num_sources = x_source.shape[0]
        self_loop_index = torch.arange(num_sources, device=x_source.device)
        edge_index_ss = torch.stack([self_loop_index, self_loop_index], dim=0)

        for hop in self.hops:
            h_s = x_source
            path = self.source_paths[f'hop_{hop}']
            for i, conv in enumerate(path):
                h_s = conv(h_s, edge_index_ss)
                if i < len(path) - 1:
                    h_s = F.leaky_relu(h_s)
                    h_s = self.dropout(h_s)
            source_hop_outputs.append(h_s)

        h_source = torch.cat(source_hop_outputs, dim=-1)
        h_source = self.source_norm(h_source)

        h_receptor_final = h_receptor
        attention_weights = None

        if edge_index_rs.shape[1] > 0:
            edge_index_sr = torch.stack([edge_index_rs[1], edge_index_rs[0]], dim=0)

            # Extract attention weights if requested
            if return_attention_weights:
                h_receptor_cross, (edge_index_with_weights, alpha) = self.cross_conv_s2r(
                    (h_source, h_receptor), edge_index_sr, edge_attr=edge_attr_rs,
                    return_attention_weights=True
                )
                attention_weights = (edge_index_with_weights, alpha)
            else:
                h_receptor_cross = self.cross_conv_s2r((h_source, h_receptor), edge_index_sr, edge_attr=edge_attr_rs)

            h_receptor_final = torch.cat([h_receptor, h_receptor_cross], dim=-1)
            h_receptor_final = self.receptor_out_linear(h_receptor_final)
            h_receptor_final = self.final_receptor_norm(h_receptor_final)
            h_receptor_final = F.leaky_relu(h_receptor_final)
        else:
            h_receptor_final = self.receptor_out_linear(torch.cat([h_receptor, torch.zeros_like(h_receptor)], dim=-1))
            h_receptor_final = self.final_receptor_norm(h_receptor_final)
            h_receptor_final = F.leaky_relu(h_receptor_final)

        h_source_final = self.final_source_norm(self.source_out_linear(h_source))
        h_source_final = F.leaky_relu(h_source_final)

        if return_attention_weights:
            return h_receptor_final, h_source_final, attention_weights
        else:
            return h_receptor_final, h_source_final

class LinkDecoder(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, activation: str = 'sigmoid'):
        super().__init__()
        self.activation_name = activation.lower()

        self.backbone = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_channels // 2, 7)
        )
        if self.activation_name == 'softplus':
            self.final_act = nn.Softplus()
        else:
            self.final_act = nn.Sigmoid()

        self._initialize_for_uniform_output()

    def _initialize_for_uniform_output(self):
        with torch.no_grad():
            if hasattr(self.backbone[-1], 'weight'):
                nn.init.normal_(self.backbone[-1].weight, mean=0.0, std=0.01)
            if hasattr(self.backbone[-1], 'bias') and self.backbone[-1].bias is not None:
                self.backbone[-1].bias.fill_(0.0)

    def forward(self, z_receptor: torch.Tensor, z_source: torch.Tensor) -> torch.Tensor:
        z = torch.cat([z_receptor, z_source], dim=-1)
        y = self.backbone(z)
        output = self.final_act(y)
        return output

class PhysGAT(nn.Module):
    def __init__(self, receptor_in_channels: int, source_in_channels: int,
                 hidden_channels: int, out_channels: int, num_heads: int,
                 dropout_rate: float, hops: list[int], **kwargs):
        super().__init__()
        self.encoder = PhysGATModel(
            receptor_in_channels=receptor_in_channels,
            source_in_channels=source_in_channels,
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            num_heads=num_heads,
            dropout_rate=dropout_rate,
            hops=hops,
            **kwargs
        )
        self.decoder = LinkDecoder(in_channels=out_channels * 2, hidden_channels=hidden_channels,
                                   activation=kwargs.get('decoder_activation', 'sigmoid'))

    def forward(self, x_receptor, x_source, edge_index_rr, edge_index_rs, edge_attr_rr=None, edge_attr_rs=None, return_attention_weights=False):
        # Call encoder with return_attention_weights parameter
        encoder_output = self.encoder(x_receptor, x_source, edge_index_rr, edge_index_rs, edge_attr_rr, edge_attr_rs, return_attention_weights=return_attention_weights)

        if return_attention_weights:
            h_receptor, h_source, attention_weights = encoder_output
        else:
            h_receptor, h_source = encoder_output
            attention_weights = None

        receptor_indices = edge_index_rs[1]
        source_indices = edge_index_rs[0]

        z_receptor = h_receptor[receptor_indices]
        z_source = h_source[source_indices]

        contributions = self.decoder(z_receptor, z_source)

        if return_attention_weights:
            return contributions, attention_weights
        else:
            return contributions

    def enable_dropout(self):
        for m in self.modules():
            if isinstance(m, nn.Dropout):
                m.train()

    def disable_dropout(self):
        for m in self.modules():
            if isinstance(m, nn.Dropout):
                m.eval()