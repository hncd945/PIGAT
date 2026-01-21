#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Model trainer module for PhysGAT training, evaluation, and prediction workflows.
Handles loss computation, optimization, early stopping, and contribution prediction.

PhysGAT模型训练模块，用于训练、评估和预测工作流程。
处理损失计算、优化器、早停机制和贡献度预测。

Author: Wenhao Wang
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_scatter import scatter_sum
from tqdm import tqdm
import logging
import numpy as np
import pandas as pd
import os
import csv
from datetime import datetime
from typing import Dict, Any, Tuple

try:
    from .model import PhysGATModel, LinkDecoder
except ImportError:
    from model import PhysGATModel, LinkDecoder
from sklearn.model_selection import train_test_split

class Trainer:
    """A Trainer class to handle the training, evaluation, and prediction process."""
    _global_acc_data_checked = False
    _global_acc_data_available = False

    def __init__(self, model: PhysGATModel, decoder: LinkDecoder, config: Dict[str, Any], dataset):
        self.config = config
        self.device = torch.device(config.device if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.decoder = decoder.to(self.device)
        self.receptors_df = dataset.receptors_df
        self.sources_df = dataset.sources_df

        # Adaptive weight system using learnable parameters
        self.log_vars = torch.nn.Parameter(torch.zeros(4, device=self.device))  # 4 loss components
        self.loss_component_history = {}

        # Track adaptive weights history for visualization
        self.adaptive_weights_history = {
            'chemistry': [],
            'distance': [],
            'strength': [],
            'reconstruction': []
        }

        # Loss function CSV recording system (per-component CSVs)
        self.loss_csv_paths = {}
        self.initial_losses = None
        self.convergence_target = 5.0

        # Controls whether this trainer writes loss CSVs (enabled only for representative model in ensemble)
        self.enable_csv_logging = True

        self.parameters = list(self.model.parameters()) + list(self.decoder.parameters()) + [self.log_vars]

        self.optimizer = torch.optim.AdamW(
            self.parameters,
            lr=config.training.learning_rate * 2.0,
            weight_decay=1e-5,
            betas=(0.9, 0.999),
            eps=1e-8
        )

        self.base_lr = float(self.config.training.learning_rate) * 2.0

        # Configure step-wise learning rate scheduler for stable training
        step_size = int(getattr(config.training, 'step_size', 50))  # Minimum 50 epochs between adjustments
        step_gamma = float(getattr(config.training, 'step_gamma', 0.8))  # Step-wise decay factor
        plateau_min_lr = float(getattr(config.training, 'plateau_min_lr', 1e-6))

        # Use StepLR for step-wise decay pattern
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=step_size,
            gamma=step_gamma
        )

        # Store minimum learning rate for manual checking
        self.min_lr = plateau_min_lr

        # Track last adjustment epoch for minimum interval enforcement
        self.last_lr_adjustment_epoch = 0
        self.min_lr_interval = step_size

        self.use_amp = config.training.amp and self.device.type == 'cuda'
        self.scaler = torch.amp.GradScaler(self.device.type) if self.use_amp else torch.amp.GradScaler('cpu')
        logging.info(f"Trainer initialized to run on '{self.device.type}'. AMP enabled: {self.use_amp}")

        self.loss_component_history = {
            'chemistry': [],
            'distance': [],
            'strength': [],
            'reconstruction': []
        }
        self.lr_history = []

        try:
            import pysheds
            import rasterio
            self.grid = None
            self.dem_raster = None
        except ImportError:
            logging.error(f"Failed to initialize pysheds/rasterio models", exc_info=True)
            self.grid = None
            self.dem_raster = None

    def _prepare_edge_attributes(self, data: Data):
        """Prepare edge attributes for the model.

        Edge attributes (7-dimensional vector):
        [0] Distance (DEM-based for atmosphere, Euclidean for others)
        [1] Wind direction alignment (for atmosphere sources, 0.5 for others)
        [2] Slope angle (for irrigation sources in degrees, 0 for others)
        [3-6] Reserved for future features (currently set to 1.0)
        """
        if hasattr(data, 'edge_attr_rs_prepared') and data.edge_attr_rs_prepared:
            return

        receptor_indices, source_indices = data.edge_index_rs
        num_edges = receptor_indices.shape[0]

        # Extract distance features
        distances = None
        if hasattr(data, 'edge_attr_rs') and data.edge_attr_rs is not None and data.edge_attr_rs.shape[0] == num_edges and data.edge_attr_rs.shape[1] >= 1:
            distances = data.edge_attr_rs[:, 0]
        elif hasattr(data, 'dem_distances') and data.dem_distances is not None and len(data.dem_distances) == num_edges:
            distances = data.dem_distances
            logging.info("Using DEM-based distances for edge attributes (edge_attr_rs[:,0]).")
        else:
            distances = torch.zeros(num_edges, dtype=torch.float, device=self.device)
            logging.warning("edge_attr_rs distances unavailable; falling back to zeros. Distance loss will remain ~1 until distances are provided.")

        distances = distances.to(self.device).float()

        # Extract wind direction alignment features
        wind_alignment = None
        if hasattr(data, 'wind_features') and data.wind_features is not None and len(data.wind_features) == num_edges:
            wind_alignment = data.wind_features
            logging.info("Using wind direction alignment features for edge attributes (edge_attr_rs[:,1]).")
        else:
            wind_alignment = torch.full((num_edges,), 0.5, dtype=torch.float, device=self.device)
            logging.info("Wind features unavailable; using default value 0.5 (neutral alignment).")

        wind_alignment = wind_alignment.to(self.device).float()

        # Extract slope angle features for irrigation sources
        slope_angles = None
        if hasattr(data, 'slope_angles') and data.slope_angles is not None and data.slope_angles.shape[0] == num_edges:
            slope_angles = data.slope_angles
            logging.info("Using slope angle features for edge attributes (edge_attr_rs[:,2]).")
        else:
            slope_angles = torch.zeros(num_edges, dtype=torch.float, device=self.device)
            logging.info("Slope angle features unavailable; using default value 0.0.")

        slope_angles = slope_angles.to(self.device).float()

        # Construct 7-dimensional edge attribute vector
        data.edge_attr_rs = torch.stack([
            distances,              # [0] Distance feature
            wind_alignment,         # [1] Wind direction alignment feature
            slope_angles,           # [2] Slope angle feature (for irrigation sources)
            torch.ones_like(distances),  # [3] Reserved
            torch.ones_like(distances),  # [4] Reserved
            torch.ones_like(distances),  # [5] Reserved
            torch.ones_like(distances),  # [6] Reserved
        ], dim=1)

        data.edge_attr_rs_prepared = True

    def train_step(self, data: Data, epoch: int = 0) -> tuple:
        self._prepare_edge_attributes(data)

        z_receptor, z_source = self.model(
            data.x_receptor, data.x_source,
            data.edge_index_rr, data.edge_index_rs,
            getattr(data, 'edge_attr_rr', None), getattr(data, 'edge_attr_rs', None)
        )

        receptor_indices, source_indices = data.edge_index_rs

        absolute_contributions = self.decoder(z_receptor[receptor_indices], z_source[source_indices])
        absolute_contributions = torch.relu(absolute_contributions)

        loss_components = self._calculate_all_losses(
            data, absolute_contributions, receptor_indices, source_indices
        )

        # Add sparsity regularization (entropy + Gini) to prevent uniform distribution
        sparsity_reg = self._calculate_entropy_regularization(
            absolute_contributions, receptor_indices, source_indices
        )
        loss_components['sparsity_regularization'] = sparsity_reg

        # Log sparsity regularization every 50 epochs
        if epoch == 1 or epoch % 50 == 0:
            logging.info(f"[Epoch {epoch}] Sparsity Regularization (Entropy + Gini): {sparsity_reg.item():.6f}")

        # Apply loss normalization for balanced optimization
        normalized_losses = self._normalize_loss_components(loss_components, epoch)

        # Calculate adaptive weights using learnable log_vars
        adaptive_weights = self._calculate_adaptive_weights()

        # Record adaptive weights for visualization
        self._record_adaptive_weights(adaptive_weights, epoch)

        total_loss = (
            adaptive_weights['chemistry'] * normalized_losses['chemistry'] +
            adaptive_weights['distance'] * normalized_losses['distance'] +
            adaptive_weights['strength'] * normalized_losses['strength'] +
            adaptive_weights['reconstruction'] * normalized_losses['reconstruction'] +
            sparsity_reg  # Add sparsity regularization (entropy + Gini) directly (not normalized)
        )

        for key in ['chemistry', 'distance', 'strength', 'reconstruction']:
            self.loss_component_history[key].append(loss_components[key].item())

        return total_loss, loss_components

    def _normalize_loss_components(self, loss_components: Dict[str, torch.Tensor], epoch: int) -> Dict[str, torch.Tensor]:
        """Normalize loss components to have similar initial scales for balanced optimization."""

        # Record initial loss values for normalization (first epoch only)
        if not hasattr(self, 'loss_normalization_factors'):
            if epoch == 1:
                # Target initial loss scale (all losses should start around this value)
                target_initial_scale = 1.0

                self.loss_normalization_factors = {
                    'chemistry': target_initial_scale / max(loss_components['chemistry'].item(), 1e-8),
                    'distance': target_initial_scale / max(loss_components['distance'].item(), 1e-8),
                    'strength': target_initial_scale / max(loss_components['strength'].item(), 1e-8),
                    'reconstruction': target_initial_scale / max(loss_components['reconstruction'].item(), 1e-8)
                }

                logging.info("=== Loss Normalization Factors ===")
                for key, factor in self.loss_normalization_factors.items():
                    original_val = loss_components[key].item()
                    normalized_val = original_val * factor
                    logging.info(f"{key}: {original_val:.6f} -> {normalized_val:.6f} (factor: {factor:.6f})")
            else:
                # Use default factors if not initialized
                self.loss_normalization_factors = {
                    'chemistry': 1.0, 'distance': 1.0, 'strength': 1.0, 'reconstruction': 1.0
                }

        # Apply normalization factors
        normalized_losses = {}
        for key in ['chemistry', 'distance', 'strength', 'reconstruction']:
            factor = self.loss_normalization_factors.get(key, 1.0)
            normalized_losses[key] = loss_components[key] * factor

        return normalized_losses

    def _calculate_adaptive_weights(self) -> Dict[str, torch.Tensor]:
        """Calculate adaptive weights using constrained task-importance weighting.

        FIXED: Replaced uncertainty weighting with constrained importance-based weighting.
        Key improvements:
        1. Chemistry + Reconstruction weights >= chemical_total_min (ensure chemical fingerprint reconstruction priority)
        2. Individual weight bounds [min, max] (prevent single loss dominance)
        3. Task importance based weighting instead of uncertainty weighting

        All constraints are now configurable via loss_weights.adaptive_weights in config.
        """
        # Calculate raw weights using learnable parameters
        weights_raw = torch.exp(-self.log_vars)
        weights_unconstrained = torch.softmax(weights_raw, dim=0)

        # Get adaptive weight constraints from config
        adaptive_cfg = self.config.loss_weights.adaptive_weights

        # Apply constraints to ensure chemical fingerprint reconstruction priority
        chemistry_weight = torch.clamp(
            weights_unconstrained[0],
            min=adaptive_cfg.chemistry.get('min', 0.20),
            max=adaptive_cfg.chemistry.get('max', 0.35)
        )
        distance_weight = torch.clamp(
            weights_unconstrained[1],
            min=adaptive_cfg.distance.get('min', 0.15),
            max=adaptive_cfg.distance.get('max', 0.30)
        )
        strength_weight = torch.clamp(
            weights_unconstrained[2],
            min=adaptive_cfg.strength.get('min', 0.15),
            max=adaptive_cfg.strength.get('max', 0.30)
        )
        reconstruction_weight = torch.clamp(
            weights_unconstrained[3],
            min=adaptive_cfg.reconstruction.get('min', 0.20),
            max=adaptive_cfg.reconstruction.get('max', 0.35)
        )

        # Ensure chemistry + reconstruction >= chemical_total_min (chemical fingerprint priority)
        chemical_total_min = adaptive_cfg.get('chemical_total_min', 0.5)
        chemical_total = chemistry_weight + reconstruction_weight
        if chemical_total < chemical_total_min:
            # Boost chemical-related weights proportionally
            boost_factor = chemical_total_min / chemical_total
            chemistry_weight = chemistry_weight * boost_factor
            reconstruction_weight = reconstruction_weight * boost_factor

            # Adjust other weights to maintain sum = 1.0
            remaining = 1.0 - chemistry_weight - reconstruction_weight
            distance_weight = distance_weight * (remaining / (distance_weight + strength_weight))
            strength_weight = strength_weight * (remaining / (distance_weight + strength_weight))

        # Final normalization to ensure sum = 1.0
        total = chemistry_weight + distance_weight + strength_weight + reconstruction_weight
        chemistry_weight = chemistry_weight / total
        distance_weight = distance_weight / total
        strength_weight = strength_weight / total
        reconstruction_weight = reconstruction_weight / total

        return {
            'chemistry': chemistry_weight,
            'distance': distance_weight,
            'strength': strength_weight,
            'reconstruction': reconstruction_weight
        }

    def _record_adaptive_weights(self, weights: Dict[str, torch.Tensor], epoch: int):
        """Record adaptive weights for visualization."""
        for key in ['chemistry', 'distance', 'strength', 'reconstruction']:
            weight_val = weights[key].item() if hasattr(weights[key], 'item') else float(weights[key])
            self.adaptive_weights_history[key].append(weight_val)

        # Log weights every 10 epochs for monitoring
        if epoch % 10 == 0:
            weight_str = ", ".join([f"{k}: {v.item():.4f}" for k, v in weights.items()])
            logging.info(f"[EPOCH {epoch}] Adaptive weights: {weight_str}")

    def _calculate_all_losses(self, data: Data, absolute_contributions: torch.Tensor,
                             receptor_indices: torch.Tensor, source_indices: torch.Tensor) -> Dict[str, float]:

        net_pollution = data.x_receptor_orig - data.unified_backgrounds  # [num_receptors, 7]

        source_fingerprints = data.x_source  # [num_sources, 7]

        chemistry_loss = self._calculate_chemistry_loss(
            absolute_contributions, receptor_indices, source_indices,
            net_pollution, source_fingerprints
        )

        distance_loss = self._calculate_distance_loss(
            absolute_contributions, receptor_indices, source_indices,
            net_pollution, data.edge_attr_rs
        )

        strength_loss = self._calculate_strength_loss(
            absolute_contributions, receptor_indices, source_indices,
            net_pollution, source_fingerprints
        )

        reconstruction_loss = self._calculate_reconstruction_loss(
            absolute_contributions, receptor_indices, source_indices,
            net_pollution
        )

        return {
            'chemistry': chemistry_loss,
            'distance': distance_loss,
            'strength': strength_loss,
            'reconstruction': reconstruction_loss
        }

    def train(self, data: Data) -> Dict[str, Any]:
        """Train the model"""
        self.model.train()
        self.decoder.train()

        best_loss = float('inf')
        patience_counter = 0
        training_history = {k: [] for k in ['total_loss', 'chemical_loss', 'strength_loss', 'distance_loss', 'reconstruction_loss', 'learning_rate']}
        epochs_iter = tqdm(range(1, self.config.training.epochs + 1), desc="Training Progress", ncols=100)

        for epoch in epochs_iter:
            self.optimizer.zero_grad()
            device_type = self.device.type
            with torch.amp.autocast(device_type, enabled=self.use_amp):
                loss, loss_components = self.train_step(data, epoch)

            if torch.isnan(loss):
                logging.warning(f"NaN loss detected at epoch {epoch}, stopping training")
                break

            # Early stopping
            if loss.item() < best_loss:
                best_loss = loss.item()
                patience_counter = 0
            else:
                patience_counter += 1

            # Record initial loss values (first epoch)
            if epoch == 1:
                self.initial_losses = {
                    'chemistry': loss_components['chemistry'].item(),
                    'distance': loss_components['distance'].item(),
                    'strength': loss_components['strength'].item(),
                    'reconstruction': loss_components['reconstruction'].item()
                }
                logging.info(f"=== Recording initial loss values ===")
                for key, value in self.initial_losses.items():
                    logging.info(f"  Initial {key}: {value:.6f}")
                self._initialize_loss_csv()

            # Record to CSV every 10 epochs
            if epoch % 10 == 0:
                self._record_loss_to_csv(epoch, loss_components)

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.parameters, max_norm=1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            # Fix: Strict step-wise learning rate scheduling ONLY
            # Remove adaptive LR to ensure pure step-wise pattern
            current_lr = self.optimizer.param_groups[0]['lr']

            # StepLR automatically steps every step_size epochs
            # We only need to call step() once per epoch and log when it actually changes
            old_lr = current_lr
            self.scheduler.step()
            new_lr = self.optimizer.param_groups[0]['lr']

            # Log only when LR actually changes (step-wise pattern)
            if new_lr != old_lr:
                logging.info(f"[StepLR] LR step down: {old_lr:.3e} -> {new_lr:.3e} (epoch {epoch})")

            # Ensure minimum LR constraint
            if new_lr < self.min_lr:
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = self.min_lr
                logging.info(f"[StepLR] LR clamped to minimum: {self.min_lr:.3e} (epoch {epoch})")

            # Record detailed loss history
            current_lr = self.optimizer.param_groups[0]['lr']
            training_history['total_loss'].append(loss.item())
            training_history['chemical_loss'].append(loss_components['chemistry'].item())
            training_history['strength_loss'].append(loss_components['strength'].item())
            training_history['distance_loss'].append(loss_components['distance'].item())
            training_history['reconstruction_loss'].append(loss_components['reconstruction'].item())
            training_history['learning_rate'].append(current_lr)

            # Also record to instance variables for ensemble use
            self.lr_history.append(current_lr)
            epochs_iter.set_postfix({"Loss": f"{loss.item():.4f}"})

            # Early stopping check
            if patience_counter >= self.config.training.early_stopping_patience:
                logging.info(f"\nEarly stopping at epoch {epoch}, best loss: {best_loss:.6f}")
                break

        # Save adaptive weights history for visualization
        training_history['adaptive_weights'] = self.adaptive_weights_history

        return {'training_history': training_history}



    def _calculate_chemistry_loss(self, absolute_contributions: torch.Tensor,
                                 receptor_indices: torch.Tensor, source_indices: torch.Tensor,
                                 net_pollution: torch.Tensor, source_fingerprints: torch.Tensor) -> torch.Tensor:

        device = absolute_contributions.device
        num_receptors = net_pollution.shape[0]

        total_contributions = torch.zeros(num_receptors, 7, device=device, dtype=absolute_contributions.dtype)
        total_contributions.index_add_(0, receptor_indices, absolute_contributions)

        chemistry_loss = 0.0
        valid_receptors = 0

        for r_idx in range(num_receptors):
            edge_mask = (receptor_indices == r_idx)
            if not edge_mask.any():
                continue

            valid_receptors += 1
            receptor_net_pollution = net_pollution[r_idx]  # [7]
            receptor_sources = source_indices[edge_mask]
            receptor_contributions = absolute_contributions[edge_mask]  # [num_edges_for_receptor, 7]

            receptor_fingerprint = receptor_net_pollution.unsqueeze(0)  # [1, 7]
            source_fps = source_fingerprints[receptor_sources]  # [num_sources_for_receptor, 7]

            similarities = F.cosine_similarity(receptor_fingerprint, source_fps, dim=1)  # [num_sources_for_receptor]
            similarities = torch.clamp(similarities, min=0.0)

            if similarities.sum() > 1e-8:
                expected_ratios = similarities / similarities.sum()
            else:
                expected_ratios = torch.ones_like(similarities) / len(similarities)

            total_contribution_sum = receptor_contributions.sum()
            if total_contribution_sum > 1e-8:
                actual_ratios = receptor_contributions.sum(dim=1) / total_contribution_sum
            else:
                actual_ratios = torch.ones_like(expected_ratios) / len(expected_ratios)

            kl_loss = F.kl_div(
                torch.log(actual_ratios + 1e-8),
                expected_ratios,
                reduction='sum'
            )
            chemistry_loss += kl_loss

        if valid_receptors > 0:
            chemistry_loss = chemistry_loss / valid_receptors

        # Scale chemistry loss using configurable factor
        # Default: 1.1 * 3 = 3.3 (enhanced by 3x to prevent contribution averaging)
        scale_factor = self.config.loss_weights.scale_factors.get('chemistry', 3.3)
        return chemistry_loss * scale_factor

    def _calculate_distance_loss(self, absolute_contributions: torch.Tensor,
                                receptor_indices: torch.Tensor, source_indices: torch.Tensor,
                                net_pollution: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        """
        FIXED: Redesigned distance loss using KL divergence for proper convergence.

        Key improvements:
        1. Can converge to 0 when distance constraints are perfectly satisfied
        2. Consistent with other loss functions design philosophy
        3. Maintains physical meaning of inverse distance weighting

        Previous design flaws:
        - term2 had fixed minimum value of 1.0, preventing convergence to 0
        - term1 cross-entropy ratio design also prevented reaching 0
        - Inconsistent with chemistry/strength/reconstruction losses that can reach 0
        """
        num_receptors = net_pollution.shape[0]
        distances = edge_attr[:, 0]

        distance_loss = 0.0
        valid_receptors = 0

        gamma = 4.0  # Inverse distance power
        eps = 1e-6

        for r_idx in range(num_receptors):
            edge_mask = (receptor_indices == r_idx)
            if not edge_mask.any():
                continue

            receptor_distances = distances[edge_mask]
            receptor_contributions = absolute_contributions[edge_mask]
            n_sources = receptor_contributions.shape[0]
            if n_sources <= 1:
                continue

            valid_receptors += 1

            # Calculate predicted distribution
            source_contributions = receptor_contributions.sum(dim=1)
            p = torch.softmax(source_contributions, dim=0)  # Predicted distribution

            # Calculate distance-based prior distribution
            inv_distances = torch.pow(1.0 / (receptor_distances + eps), gamma)
            q = torch.softmax(inv_distances, dim=0)  # Distance prior

            # KL divergence: KL(p || q) = sum(p * log(p/q))
            # When p perfectly matches q, KL divergence = 0
            kl_loss = torch.nn.functional.kl_div(
                torch.log(p + eps), q, reduction='sum'
            )

            distance_loss += kl_loss

        if valid_receptors > 0:
            distance_loss = distance_loss / valid_receptors

        # Scale distance loss using configurable factor
        # Default: 1.7 (scale to target initial value ~1.7)
        scale_factor = self.config.loss_weights.scale_factors.get('distance', 1.7)
        return distance_loss * scale_factor

    def _calculate_strength_loss(self, absolute_contributions: torch.Tensor,
                                receptor_indices: torch.Tensor, source_indices: torch.Tensor,
                                net_pollution: torch.Tensor, source_fingerprints: torch.Tensor) -> torch.Tensor:

        num_receptors = net_pollution.shape[0]

        strength_loss = 0.0
        valid_receptors = 0

        for r_idx in range(num_receptors):
            edge_mask = (receptor_indices == r_idx)
            if not edge_mask.any():
                continue

            valid_receptors += 1
            receptor_sources = source_indices[edge_mask]
            receptor_contributions = absolute_contributions[edge_mask]  # [num_edges_for_receptor, 7]

            source_fps = source_fingerprints[receptor_sources]  # [num_sources_for_receptor, 7]
            source_strengths = torch.norm(source_fps, p=2, dim=1)  # [num_sources_for_receptor]

            source_strengths = source_strengths / (source_strengths.mean() + 1e-8)


            if source_strengths.sum() > 1e-8:
                expected_ratios = source_strengths / source_strengths.sum()
            else:
                expected_ratios = torch.ones_like(source_strengths) / len(source_strengths)

            total_contribution_sum = receptor_contributions.sum()
            if total_contribution_sum > 1e-8:
                actual_ratios = receptor_contributions.sum(dim=1) / total_contribution_sum
            else:
                actual_ratios = torch.ones_like(expected_ratios) / len(expected_ratios)

            kl_loss = F.kl_div(
                torch.log(actual_ratios + 1e-8),
                expected_ratios,
                reduction='sum'
            )
            strength_loss += kl_loss

        if valid_receptors > 0:
            strength_loss = strength_loss / valid_receptors

        # Scale strength loss using configurable factor
        # Default: 3.8 * 3 = 11.4 (enhanced by 3x to prevent contribution averaging)
        scale_factor = self.config.loss_weights.scale_factors.get('strength', 11.4)
        return strength_loss * scale_factor
    def _calculate_reconstruction_loss(self, absolute_contributions: torch.Tensor,
                                      receptor_indices: torch.Tensor, source_indices: torch.Tensor,
                                      net_pollution: torch.Tensor) -> torch.Tensor:
        """Reconstruction loss between reconstructed and net pollution fingerprints.
        The loss is defined as 1 - cosine_similarity, averaged over receptors.
        This scales the initial loss around ~1.0 for proper balancing with others.
        """
        device = absolute_contributions.device
        num_receptors = net_pollution.shape[0]

        reconstruction_loss = 0.0
        valid_receptors = 0

        # Sum absolute contribution vectors per receptor to get reconstructed fingerprint (7-dim)
        for r_idx in range(num_receptors):
            edge_mask = (receptor_indices == r_idx)
            if not edge_mask.any():
                continue

            # Reconstructed fingerprint: sum of edge contribution vectors for this receptor
            recon_fp = absolute_contributions[edge_mask].sum(dim=0)  # [7]
            true_fp = net_pollution[r_idx]  # [7]

            # Handle zero vectors to avoid NaNs in cosine similarity
            recon_norm = torch.norm(recon_fp, p=2)
            true_norm = torch.norm(true_fp, p=2)
            if recon_norm <= 1e-12 or true_norm <= 1e-12:
                sim = torch.tensor(0.0, device=device, dtype=absolute_contributions.dtype)
            else:
                sim = F.cosine_similarity(recon_fp.unsqueeze(0), true_fp.unsqueeze(0), dim=1).squeeze(0)
                sim = torch.clamp(sim, min=0.0, max=1.0)

            # Loss contribution: 1 - cosine_similarity
            reconstruction_loss = reconstruction_loss + (1.0 - sim)
            valid_receptors += 1

        if valid_receptors > 0:
            reconstruction_loss = reconstruction_loss / valid_receptors

        # Scale reconstruction loss using configurable factor
        # Default: 4.4 (0.39 * 4.4 ≈ 1.7, scale to target initial value ~1.7)
        scale_factor = self.config.loss_weights.scale_factors.get('reconstruction', 4.4)
        return reconstruction_loss * scale_factor

    def _calculate_entropy_regularization(self, absolute_contributions: torch.Tensor,
                                         receptor_indices: torch.Tensor,
                                         source_indices: torch.Tensor) -> torch.Tensor:
        """Calculate sparsity regularization using entropy and Gini coefficient.

        Strategy 1: Entropy Regularization
        - Formula: H(p) = -Σ(p_i * log(p_i))
        - Higher entropy → more uniform → higher penalty
        - Lower entropy → more sparse → lower penalty
        - Weight: entropy_weight = 0.3

        Strategy 2: Gini Coefficient Penalty
        - Formula: Gini = (2 * Σ(i * p_sorted[i])) / (n * Σ(p_sorted)) - (n+1)/n
        - Only penalize if Gini < 0.5 (too uniform)
        - Penalty: (0.5 - Gini)²
        - Weight: gini_weight = 0.5

        Args:
            absolute_contributions: [num_edges, 7] contribution values
            receptor_indices: [num_edges] receptor indices for each edge
            source_indices: [num_edges] source indices for each edge

        Returns:
            Total sparsity regularization loss (scalar)
        """
        num_receptors = receptor_indices.max().item() + 1
        eps = 1e-8

        # Get regularization parameters from config
        use_entropy = getattr(self.config.training, 'use_entropy_regularization', True)
        use_gini = getattr(self.config.training, 'use_gini_penalty', True)
        entropy_weight = getattr(self.config.training, 'entropy_weight', 0.3)
        gini_weight = getattr(self.config.training, 'gini_weight', 0.5)
        gini_threshold = getattr(self.config.training, 'gini_threshold', 0.5)

        total_sparsity_loss = 0.0
        valid_receptors = 0

        for r_idx in range(num_receptors):
            edge_mask = (receptor_indices == r_idx)
            if not edge_mask.any():
                continue

            receptor_contributions = absolute_contributions[edge_mask]  # [K, 7]
            n_sources = receptor_contributions.shape[0]

            if n_sources <= 1:
                continue

            # Calculate total contribution per source (sum across 7 metals)
            source_totals = receptor_contributions.sum(dim=1)  # [K]
            total_sum = source_totals.sum()

            if total_sum <= eps:
                continue

            # Normalize to get probability distribution (contribution ratios)
            contribution_ratios = source_totals / (total_sum + eps)  # [K]

            # Strategy 1: Entropy Regularization
            # H(p) = -sum(p * log(p))
            # Higher entropy = more uniform, lower entropy = more sparse
            # We ADD entropy to loss to penalize uniform distributions
            if use_entropy:
                entropy = -(contribution_ratios * torch.log(contribution_ratios + eps)).sum()
                total_sparsity_loss += entropy_weight * entropy

            # Strategy 2: Gini Coefficient Penalty
            # Gini coefficient measures inequality (0=perfect equality, 1=perfect inequality)
            # Only penalize if Gini < threshold (too uniform)
            if use_gini:
                # Sort contribution ratios in ascending order
                sorted_ratios, _ = torch.sort(contribution_ratios)

                # Calculate indices (1, 2, 3, ..., n)
                n = torch.tensor(n_sources, dtype=torch.float32, device=contribution_ratios.device)
                indices = torch.arange(1, n_sources + 1, dtype=torch.float32, device=contribution_ratios.device)

                # Gini coefficient formula:
                # Gini = (2 * Σ(i * p_sorted[i])) / (n * Σ(p_sorted)) - (n+1)/n
                gini = (2.0 * torch.sum(indices * sorted_ratios)) / (n * torch.sum(sorted_ratios) + eps) - (n + 1.0) / n

                # Only penalize if Gini < threshold (too uniform)
                if gini < gini_threshold:
                    gini_penalty = (gini_threshold - gini) ** 2
                    total_sparsity_loss += gini_weight * gini_penalty

            valid_receptors += 1

        if valid_receptors > 0:
            total_sparsity_loss = total_sparsity_loss / valid_receptors

        return total_sparsity_loss


    def predict(self, data: Data) -> pd.DataFrame:
        self.model.eval()
        self.decoder.eval()
        data = data.to(self.device)
        if not hasattr(data, 'edge_attr_rs_prepared') or not data.edge_attr_rs_prepared:
            self._prepare_edge_attributes(data)
        with torch.no_grad():
            z_receptor, z_source = self.model(data.x_receptor, data.x_source, data.edge_index_rr, data.edge_index_rs, getattr(data, 'edge_attr_rr', None), getattr(data, 'edge_attr_rs', None))
            receptor_indices, source_indices = data.edge_index_rs
            contributions = self.decoder(z_receptor[receptor_indices], z_source[source_indices])
        total_contributions = contributions.sum(dim=1)
        return pd.DataFrame({'receptor_idx': receptor_indices.cpu().numpy(), 'source_idx': source_indices.cpu().numpy(), 'contribution': total_contributions.cpu().numpy()})

    def _adaptive_lr_adjust(self, epoch: int):
        try:
            # Use a longer window and stronger cooldown to avoid dense micro-steps
            window = int(getattr(self.config.training, 'adaptive_lr_window', 8))
            if not hasattr(self, 'loss_component_history'):
                return

            chem_hist = self.loss_component_history.get('chemistry', [])
            dist_hist = self.loss_component_history.get('distance', [])
            str_hist  = self.loss_component_history.get('strength', [])
            recon_hist = self.loss_component_history.get('reconstruction', [])
            min_len = min(len(chem_hist), len(dist_hist), len(str_hist), len(recon_hist))
            if min_len < max(3, window):
                return

            # Build composite total loss history from components (shared-LR case)
            total_hist = [
                (chem_hist[i] + dist_hist[i] + str_hist[i] + recon_hist[i]) / 4.0
                for i in range(-min_len, 0)
            ]

            # Determine if optimizer has named param groups for per-component LRs
            group_names = [g['name'] for g in self.optimizer.param_groups if 'name' in g]
            comp_to_group = {g['name']: g for g in self.optimizer.param_groups if 'name' in g}
            has_separate = any(n in ('chemistry', 'distance', 'strength', 'reconstruction') for n in group_names)

            # Smoother but more meaningful adjustments
            cooldown = int(getattr(self.config.training, 'adaptive_lr_cooldown', 50))
            alpha = float(getattr(self.config.training, 'adaptive_lr_ema_alpha', 0.08))
            min_improve = float(getattr(self.config.training, 'adaptive_lr_min_improve', 0.02))
            # Larger step to make changes visible (default 0.93, configurable)
            down_factor = float(getattr(self.config.training, 'adaptive_lr_down_factor', 0.93))

            # Initialize caches
            if not hasattr(self, '_ema_cache'):
                self._ema_cache = {}
            if not hasattr(self, '_lr_last_adjust_epoch_group'):
                self._lr_last_adjust_epoch_group = {}

            def should_reduce(series, key):
                # EMA smoothing
                ema_key = f'ema_{key}'
                last = series[-1]
                prev1 = series[-2]
                prev2 = series[-3]
                ema_prev = self._ema_cache.get(ema_key, last)
                ema_now = alpha * last + (1.0 - alpha) * ema_prev
                self._ema_cache[ema_key] = ema_now
                # Windowed relative improvement
                p0 = series[-window]
                pN = last
                rel_improve = (p0 - pN) / (abs(p0) + 1e-8)
                trend_up = (last > prev1 >= prev2)
                return (rel_improve < min_improve) or trend_up

            if has_separate:
                # Per-component LR adaptation if param groups are provided and named
                comp_series = {
                    'chemistry': chem_hist,
                    'distance': dist_hist,
                    'strength': str_hist,
                    'reconstruction': recon_hist,
                }
                for comp, series in comp_series.items():
                    if comp not in comp_to_group:
                        continue
                    if len(series) < max(3, window):
                        continue
                    last_epoch = self._lr_last_adjust_epoch_group.get(comp, 0)
                    if epoch - last_epoch < cooldown:
                        continue
                    if should_reduce(series, comp):
                        group = comp_to_group[comp]
                        lr = group['lr']
                        min_lr = float(getattr(self.config.training, 'min_lr', 1e-6))
                        new_lr = max(lr * down_factor, min_lr)
                        if new_lr < lr - 1e-12:
                            group['lr'] = new_lr
                            self._lr_last_adjust_epoch_group[comp] = epoch
                            logging.info(f"[AdaptiveLR:{comp}] LR down: {lr:.3e} -> {new_lr:.3e} (window={window}, cooldown={cooldown})")
            else:
                # Shared LR: adapt based on composite total loss
                last_epoch = self._lr_last_adjust_epoch_group.get('global', 0)
                if epoch - last_epoch < cooldown:
                    return
                if should_reduce(total_hist, 'total'):
                    group = self.optimizer.param_groups[0]
                    lr = group['lr']
                    min_lr = float(getattr(self.config.training, 'min_lr', 1e-6))
                    new_lr = max(lr * down_factor, min_lr)
                    if new_lr < lr - 1e-12:
                        group['lr'] = new_lr
                        self._lr_last_adjust_epoch_group['global'] = epoch
                        logging.info(f"[AdaptiveLR] LR down: {lr:.3e} -> {new_lr:.3e} (composite, window={window}, cooldown={cooldown})")
        except Exception as e:
            logging.debug(f"Adaptive LR adjust skipped: {e}")

    def _initialize_loss_csv(self):
        """Initialize per-component loss CSV files: chemistry, distance, strength, reconstruction.
        Files are deterministic within the run output directory to avoid redundant duplicates.
        Columns: Epoch, Loss, Reduction_%
        """
        # Skip when CSV logging is disabled (non-representative ensemble models)
        if not getattr(self, 'enable_csv_logging', True):
            return
        from hydra.core.hydra_config import HydraConfig
        try:
            hydra_cfg = HydraConfig.get()
            outputs_dir = hydra_cfg.runtime.output_dir
        except:
            outputs_dir = "run_outputs"

        os.makedirs(outputs_dir, exist_ok=True)

        # Define per-component CSV paths with fixed names (overwritten each run)
        components = ['chemistry', 'distance', 'strength', 'reconstruction']
        self.loss_csv_paths = {
            comp: os.path.join(outputs_dir, f"loss_{comp}.csv") for comp in components
        }

        # Create/overwrite headers
        for comp, path in self.loss_csv_paths.items():
            with open(path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['Epoch', 'Loss', 'Reduction_%'])
            logging.info(f"Initialized loss CSV: {os.path.basename(path)}")

    def _record_loss_to_csv(self, epoch: int, loss_components: dict):
        # Only record if CSV logging is enabled (representative model)
        if not getattr(self, 'enable_csv_logging', True):
            return
        if not self.loss_csv_paths or self.initial_losses is None:
            return

        # Compute reductions per component
        reductions = {}
        for key in ['chemistry', 'distance', 'strength', 'reconstruction']:
            loss_val = loss_components[key].item() if hasattr(loss_components[key], 'item') else loss_components[key]
            if self.initial_losses.get(key, 0) > 0:
                reduction = (self.initial_losses[key] - loss_val) / (self.initial_losses[key] + 1e-12) * 100
                reductions[key] = max(0.0, float(reduction))
            else:
                reductions[key] = 0.0

        # Append to each per-component CSV
        for key in ['chemistry', 'distance', 'strength', 'reconstruction']:
            path = self.loss_csv_paths.get(key)
            if not path:
                continue
            loss_val = loss_components[key].item() if hasattr(loss_components[key], 'item') else loss_components[key]
            with open(path, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([epoch, f"{loss_val:.6f}", f"{reductions[key]:.2f}%"])

        logging.info(f"[EPOCH {epoch}] Loss reduction progress:")
        for key in ['chemistry', 'distance', 'strength', 'reconstruction']:
            loss_val = loss_components[key].item() if hasattr(loss_components[key], 'item') else loss_components[key]
            logging.info(f"  {key}: {loss_val:.6f} (Reduction {reductions[key]:.2f}%)")

    def _set_loss_weights(self, loss_weights: Dict[str, float]):
        self.custom_loss_weights = loss_weights
        logging.info(f"Set custom loss weights: {loss_weights}")
