#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ensemble learning module for training multiple independent PhysGAT models.
Manages parallel model training, prediction aggregation, and uncertainty quantification.

集成学习模块，用于训练多个独立的PhysGAT模型。
管理并行模型训练、预测聚合和不确定性量化。

Author: Wenhao Wang
"""
import logging
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from tqdm import tqdm

# Local imports are inside the methods to avoid circular dependencies
# with modules that might be initialized by the main script.

class EnsembleTrainer:
    """Manages the independent training of an ensemble of models without consensus mechanisms."""

    def __init__(self, config, dataset):
        """Initializes the EnsembleTrainer.

        Args:
            config (DictConfig): The main configuration object.
            dataset (PhysGATDataset): The dataset object containing all data.
        """
        self.base_config = config
        self.dataset = dataset
        self.data = dataset[0]
        self.n_ensemble = config.ensemble.n_ensemble

        self.use_consensus_loss = False
        self.use_representation_loss = False
        self.aggregation_method = getattr(config.ensemble, 'aggregation_method', 'trimmed_mean')

        self.model_diversity_control = getattr(config.ensemble, 'model_diversity_control', False)
        self.diversity_regularization = getattr(config.ensemble, 'diversity_regularization', 0.05)
        self.std_constraint = getattr(config.ensemble, 'std_constraint', False)
        self.target_std_ratio = getattr(config.ensemble, 'target_std_ratio', 0.15)
        self.std_penalty_weight = getattr(config.ensemble, 'std_penalty_weight', 0.1)
        self.top_k_models = getattr(config.ensemble, 'top_k_models', 5)  # Second epoch aggressive strategy

        # Enhanced ensemble consistency parameters
        self.weight_allocation_consistency = getattr(config.ensemble, 'weight_allocation_consistency', False)
        self.consistency_regularization_weight = getattr(config.ensemble, 'consistency_regularization_weight', 0.08)
        self.max_weight_deviation = getattr(config.ensemble, 'max_weight_deviation', 0.25)
        self.consistency_warmup_epochs = getattr(config.ensemble, 'consistency_warmup_epochs', 50)

        # Initialize weight tracking for consistency
        self.model_weight_history = []
        self.ensemble_mean_weights = None

        self.model_predictions = []
        self.trim_percentage = getattr(config.ensemble, 'trim_percentage', 0.2)

        self.loss_masking_test = getattr(config.ensemble, 'loss_masking_test', False)
        self.enabled_losses = getattr(config.ensemble, 'enabled_losses', {
            'chemistry': True, 'strength': True,
            'distance': True, 'pearson': True, 'rank': True
        })
        self.use_uncertainty_weighting = getattr(config.ensemble, 'use_uncertainty_weighting', True)
        self.use_simple_mse = getattr(config.ensemble, 'use_simple_mse', False)

        # Model storage
        self.models = []
        self.decoders = []
        self.optimizers = []
        self.schedulers = []
        self.trainers = []

        # Results storage
        self.predictions = []
        self.model_losses = []
        self.model_weights = []
        self.representative_trainer = None  # To store the first trainer for XAI
        self.representative_history = None # To store the history of the first trainer


    def train_ensemble(self):
        """Trains an ensemble of models synchronously with knowledge distillation."""
        logging.info(f"\n=== Starting synchronous ensemble training for {self.n_ensemble} models ===")
        
        # Initialize all models, decoders, and optimizers
        self._initialize_ensemble()

        # --- Move data to the correct device before any computation ---
        device = torch.device(self.base_config.device if torch.cuda.is_available() else "cpu")
        self.data = self.data.to(device)

        # --- Prepare data attributes before training ---
        if self.n_ensemble > 0:
            logging.info("Preparing shared data object with dynamic attributes...")
            self.trainers[0]._prepare_edge_attributes(self.data)
            logging.info("Data preparation complete.")

        logging.info("=== Using uncertainty-weighted loss system ===")

        logging.info("Training models independently - all consensus mechanisms disabled")
        self._train_independently()

        # Generate predictions from all models
        self._generate_ensemble_predictions()

        self._calculate_model_weights()
        logging.info("Ensemble training complete.")

        # Extract real attention weights from the representative model
        logging.info("=" * 70)
        logging.info("Extracting REAL attention weights from trained model...")
        logging.info("=" * 70)
        self._extract_and_save_attention_weights()

        return self.get_ensemble_predictions()

    def _initialize_ensemble(self):
        """Initialize all models, decoders, optimizers, and schedulers."""
        from src.model import PhysGATModel, LinkDecoder
        from src.trainer import Trainer
        import torch.optim as optim

        logging.info("Initializing ensemble models...")

        for i in range(self.n_ensemble):
            cfg = self._get_improved_config(i)

            # Create model and decoder
            model = PhysGATModel(
                receptor_in_channels=self.data.x_receptor.shape[1],
                source_in_channels=self.data.x_source.shape[1],
                hidden_channels=cfg.model.hidden_channels,
                out_channels=cfg.model.out_channels,
                num_heads=cfg.model.num_heads,
                dropout_rate=cfg.model.dropout_rate,
                hops=cfg.model.multi_scale.hops
            )
            decoder = LinkDecoder(
                in_channels=cfg.model.out_channels * 2,
                hidden_channels=cfg.model.hidden_channels,
                activation=getattr(cfg.model, 'decoder_activation', 'sigmoid')
            )

            # Move to device
            device = torch.device(cfg.device if torch.cuda.is_available() else 'cpu')
            model = model.to(device)
            decoder = decoder.to(device)

            # Create trainer (for loss computation and other utilities)
            trainer = Trainer(model, decoder, cfg, self.dataset)
            # Only the first (representative) model writes loss CSVs to avoid overwriting across ensemble
            try:
                trainer.enable_csv_logging = (i == 0)
            except Exception:
                pass

            # Create optimizer
            params = list(model.parameters()) + list(decoder.parameters())
            optimizer = optim.Adam(params, lr=cfg.training.learning_rate,
                                 weight_decay=cfg.training.weight_decay)

            # Create scheduler
            scheduler = None
            if cfg.training.scheduler_type == "cosine_annealing":
                scheduler = optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=cfg.training.epochs
                )
            elif cfg.training.scheduler_type == "exponential":
                scheduler = optim.lr_scheduler.ExponentialLR(
                    optimizer, gamma=cfg.training.lr_decay_rate
                )
            elif cfg.training.scheduler_type == "step":
                scheduler = optim.lr_scheduler.StepLR(
                    optimizer, step_size=cfg.training.lr_step_size,
                    gamma=cfg.training.lr_gamma
                )

            # Store everything
            self.models.append(model)
            self.decoders.append(decoder)
            self.optimizers.append(optimizer)
            self.schedulers.append(scheduler)
            self.trainers.append(trainer)

            # Save the first trainer as representative
            if i == 0:
                self.representative_trainer = trainer

        logging.info(f"Initialized {self.n_ensemble} models successfully.")


    def _get_improved_config(self, model_idx):
        """Gets a varied config for each model in the ensemble."""
        config = OmegaConf.to_container(self.base_config, resolve=True)
        config['seed'] = self.base_config.seed + model_idx * 500
        base_wd = self.base_config.training.weight_decay
        config['training']['weight_decay'] = base_wd * (1.0 + 0.03 * model_idx)
        lr_multiplier = 1.0 + 0.01 * (model_idx - self.n_ensemble // 2) / self.n_ensemble
        config['training']['learning_rate'] = self.base_config.training.learning_rate * lr_multiplier

        base_patience = getattr(self.base_config.training, 'early_stopping_patience', None)
        if base_patience is not None:
            config['training']['early_stopping_patience'] = int(base_patience)
        else:
            config['training']['early_stopping_patience'] = 40 + model_idx * 2

        config['model']['dropout_rate'] = 0.1 + 0.01 * model_idx
        return OmegaConf.create(config)

    def _train_independently(self):
        """Enhanced independent training with weight allocation consistency."""
        logging.info("Training models independently with consistency constraints")

        for i, trainer in enumerate(self.trainers):
            logging.info(f"--- Training model {i+1}/{self.n_ensemble} ---")

            # Add weight consistency constraint to trainer if enabled
            if self.weight_allocation_consistency:
                trainer.ensemble_consistency_config = {
                    'enabled': True,
                    'regularization_weight': self.consistency_regularization_weight,
                    'max_deviation': self.max_weight_deviation,
                    'warmup_epochs': self.consistency_warmup_epochs,
                    'ensemble_index': i,
                    'total_models': self.n_ensemble
                }

            history = trainer.train(self.data)

            # Save the first trainer's history
            if i == 0:
                self.representative_history = history

            final_loss = min(history['training_history']['total_loss']) if history.get('training_history', {}).get('total_loss') else float('inf')
            self.model_losses.append(final_loss)
            logging.info(f"Model {i+1} finished with best loss: {final_loss:.6f}")

            # Track weight allocation patterns for consistency analysis
            if hasattr(trainer, 'weight_allocation_history'):
                self.model_weight_history.append(trainer.weight_allocation_history)

    def _generate_ensemble_predictions(self):
        """Generate predictions from all trained models."""
        logging.info("Generating predictions from ensemble models...")

        for i, trainer in enumerate(self.trainers):
            predictions = trainer.predict(self.data)
            self.predictions.append(predictions)


    pass


    def _calculate_model_weights(self):
        """Calculates model weights based on performance."""
        if not self.model_losses:
            self.model_weights = [1.0 / self.n_ensemble] * self.n_ensemble
            return
        inverse_losses = [1.0 / (loss + 1e-8) for loss in self.model_losses]
        total_weight = sum(inverse_losses)
        self.model_weights = [w / total_weight for w in inverse_losses]

    def get_ensemble_predictions(self):
        """Aggregate predictions and compute statistics on per-receptor percentage contributions.

        This fixes the std calculation method by:
        1) Computing, for each model and receptor, the percentage contribution per source (summing to 100%).
        2) Aggregating these percentages across models to get mean and std (weighted and robust when configured).
        3) Preserving absolute contribution aggregation for backward compatibility.
        """
        logging.info("\n=== Aggregating ensemble predictions ===")

        # Attach model weights and IDs
        frames = []
        n_models = len(self.predictions)
        default_w = 1.0 / max(n_models, 1)
        for i, pred_df in enumerate(self.predictions):
            df = pred_df.copy()
            df['model_weight'] = self.model_weights[i] if i < len(self.model_weights) else default_w
            df['model_id'] = i
            frames.append(df)
        all_predictions = pd.concat(frames, ignore_index=True)

        # Compute per-model, per-receptor totals
        totals = (all_predictions.groupby(['model_id', 'receptor_idx'])['contribution']
                  .sum().reset_index().rename(columns={'contribution': 'total_contribution'}))
        all_predictions = all_predictions.merge(totals, on=['model_id', 'receptor_idx'], how='left')

        # Convert to per-model percentages (0–100)
        # IMPORTANT: Do NOT force equal distribution when total is zero
        # Let the model learn natural sparse distributions
        all_predictions['contribution_percent'] = (all_predictions['contribution'] / (all_predictions['total_contribution'] + 1e-8)) * 100.0

        # Only fill NaN values with 0, do NOT force equal distribution
        all_predictions['contribution_percent'] = all_predictions['contribution_percent'].fillna(0.0)

        def robust_agg(group: pd.DataFrame) -> pd.Series:
            """Robust aggregation for both absolute and percentage values."""
            # Percent-level stats (primary)
            percents = group['contribution_percent'].values.astype(float)
            weights = group['model_weight'].values.astype(float)

            def _agg(values: np.ndarray, weights: np.ndarray):
                if self.aggregation_method == "weighted_mean":
                    w = weights ** 1.5
                    w = w / (w.sum() + 1e-12)
                    mean = np.average(values, weights=w)
                    std = np.sqrt(np.average((values - mean) ** 2, weights=w))
                elif self.aggregation_method == "trimmed_mean":
                    n_trim = int(len(values) * self.trim_percentage / 2)
                    if n_trim > 0 and len(values) > 2 * n_trim:
                        idx = np.argsort(values)
                        keep = idx[n_trim:-n_trim]
                        v = values[keep]
                        w = weights[keep]
                        w = w / (w.sum() + 1e-12)
                        mean = np.average(v, weights=w)
                        std = np.sqrt(np.average((v - mean) ** 2, weights=w))
                    else:
                        mean = np.average(values, weights=weights)
                        std = np.sqrt(np.average((values - mean) ** 2, weights=weights))
                elif self.aggregation_method == "median":
                    mean = float(np.median(values))
                    std = float(np.std(values))
                elif self.aggregation_method == "top_models_only":
                    top_k = getattr(self, 'top_k_models', 5)
                    if len(values) > top_k:
                        top_idx = np.argsort(weights)[-top_k:]
                        v = values[top_idx]
                        w = weights[top_idx]
                        w = w / (w.sum() + 1e-12)
                        mean = np.average(v, weights=w)
                        std = np.sqrt(np.average((v - mean) ** 2, weights=w))
                    else:
                        mean = np.average(values, weights=weights)
                        std = np.sqrt(np.average((values - mean) ** 2, weights=weights))
                else:  # mean
                    mean = np.average(values, weights=weights)
                    std = np.sqrt(np.average((values - mean) ** 2, weights=weights))

                # Optional std constraint refinement
                if self.std_constraint:
                    std_ratio = std / (abs(mean) + 1e-8)
                    if std_ratio > self.target_std_ratio:
                        w = weights ** 3.0
                        w = w / (w.sum() + 1e-12)
                        mean = np.average(values, weights=w)
                        std = np.sqrt(np.average((values - mean) ** 2, weights=w))
                return mean, std

            percent_mean, percent_std = _agg(percents, weights)

            # Absolute-level stats (backward compatibility)
            contributions = group['contribution'].values.astype(float)
            abs_mean, abs_std = _agg(contributions, weights)

            # Outliers removed estimate for the configured method (based on values length)
            n_trim = int(len(percents) * self.trim_percentage / 2)
            outliers_removed = 2 * n_trim if (self.aggregation_method == "trimmed_mean" and n_trim > 0 and len(percents) > 2 * n_trim) else 0

            return pd.Series({
                'contribution_mean': abs_mean,
                'contribution_std': abs_std,
                'contribution_percent_mean': percent_mean,
                'contribution_percent_std': percent_std,
                'n_models_used': len(np.unique(group['model_id'].values)),
                'outliers_removed': outliers_removed,
                'std_ratio': percent_std / (abs(percent_mean) + 1e-8)
            })

        # Aggregate per (receptor, source) across models
        ensemble_results = (all_predictions.groupby(['receptor_idx', 'source_idx'])
                             .apply(robust_agg).reset_index())

        # Verify normalization at the receptor level using percentage means
        totals = ensemble_results.groupby('receptor_idx')['contribution_percent_mean'].sum()
        logging.info(f"Ensemble percent normalization check — mean={totals.mean():.2f}%, std={totals.std():.2f}% | range=[{totals.min():.2f}%, {totals.max():.2f}%]")

        logging.info("Ensemble prediction aggregation complete.")
        return ensemble_results

    def _extract_and_save_attention_weights(self):
        """Extract REAL attention weights from the trained representative model.

        This method:
        1. Uses the trained representative model
        2. Performs a forward pass with return_attention_weights=True
        3. Extracts real attention weights from cross_conv_s2r layer
        4. Combines with contribution data and edge attributes
        5. Saves to CSV file

        NO SIMULATION, NO FAKE DATA - all data from actual model computation.
        """
        import os
        from hydra.core.hydra_config import HydraConfig

        try:
            hydra_cfg = HydraConfig.get()
            output_dir = hydra_cfg.runtime.output_dir
        except:
            output_dir = "run_outputs"

        logging.info("Starting real attention weight extraction...")
        logging.info(f"Output directory: {output_dir}")

        # Get the representative model and decoder
        if not hasattr(self, 'representative_trainer') or self.representative_trainer is None:
            logging.warning("No representative trainer found. Skipping attention weight extraction.")
            return

        model = self.representative_trainer.model
        decoder = self.representative_trainer.decoder
        device = self.data.x_receptor.device

        # Set model to evaluation mode
        model.eval()
        decoder.eval()

        logging.info("Performing forward pass to extract attention weights...")

        with torch.no_grad():
            # Forward pass through encoder (PhysGATModel) with return_attention_weights=True
            # Note: edge_attr_rr is None (not used in the model), edge_attr_rs is prepared by trainer
            h_receptor, h_source, attention_weights = model(
                self.data.x_receptor,
                self.data.x_source,
                self.data.edge_index_rr,
                self.data.edge_index_rs,
                edge_attr_rr=None,
                edge_attr_rs=self.data.edge_attr_rs if hasattr(self.data, 'edge_attr_rs') else None,
                return_attention_weights=True
            )

            # Unpack attention weights
            # edge_index_with_weights is in (source, receptor) format from cross_conv_s2r
            # It corresponds to edge_index_sr which is the transposed version of edge_index_rs
            edge_index_with_weights, alpha = attention_weights

            # edge_index_with_weights format: [source_node_ids, receptor_node_ids]
            # We need to use these indices to get embeddings and compute contributions
            source_indices = edge_index_with_weights[0]  # source node IDs
            receptor_indices = edge_index_with_weights[1]  # receptor node IDs
            z_receptor = h_receptor[receptor_indices]
            z_source = h_source[source_indices]
            contributions = decoder(z_receptor, z_source)

        logging.info(f"✓ Extracted {len(alpha)} real attention weights")
        logging.info(f"  Attention weight range: [{alpha.min().item():.4f}, {alpha.max().item():.4f}]")
        logging.info(f"  Attention weight mean: {alpha.mean().item():.4f}")
        logging.info(f"  Attention weight std: {alpha.std().item():.4f}")

        # Convert to numpy
        alpha_np = alpha.cpu().numpy()
        edge_index_np = edge_index_with_weights.cpu().numpy()
        contributions_np = contributions.cpu().numpy()
        edge_attrs_np = self.data.edge_attr_rs.cpu().numpy()

        # Debug: Log edge attribute information
        logging.info(f"Edge attributes shape: {edge_attrs_np.shape}")
        logging.info(f"Slope angle (column 2) - Min: {edge_attrs_np[:, 2].min():.4f}, Max: {edge_attrs_np[:, 2].max():.4f}, Mean: {edge_attrs_np[:, 2].mean():.4f}")
        if hasattr(self.data, 'slope_angles') and self.data.slope_angles is not None:
            logging.info(f"Original slope_angles - Min: {self.data.slope_angles.min().item():.4f}, Max: {self.data.slope_angles.max().item():.4f}, Mean: {self.data.slope_angles.mean().item():.4f}")

        # Get source and receptor dataframes
        receptors_df = self.dataset.receptors_df
        sources_df = self.dataset.sources_df

        # Build comprehensive data table
        logging.info("Building comprehensive XAI data table...")
        xai_data = []

        for i in range(len(alpha_np)):
            # Get source and receptor indices
            # Note: edge_index_with_weights is (source, receptor) format from cross_conv_s2r
            source_idx = int(edge_index_with_weights[0, i].item())
            receptor_idx = int(edge_index_with_weights[1, i].item())

            # Get source and receptor info
            source_info = sources_df.iloc[source_idx]
            receptor_info = receptors_df.iloc[receptor_idx]

            # Get edge attributes (7 dimensions)
            edge_attr = edge_attrs_np[i]

            # Calculate contribution percentage (sum across 7 metals)
            contrib_abs = contributions_np[i]  # Shape: (7,)
            contrib_sum = float(np.sum(contrib_abs))

            # Build data row
            data_row = {
                'source_idx': source_idx,
                'receptor_idx': receptor_idx,
                'source_sample': source_info['sample'],
                'receptor_sample': receptor_info['sample'],
                'source_type': source_info['source_type'],
                'source_lon': source_info['lon'],
                'source_lat': source_info['lat'],
                'receptor_lon': receptor_info['lon'],
                'receptor_lat': receptor_info['lat'],
                'attention_weight': float(alpha_np[i]),
                'predicted_contribution_sum': contrib_sum,
                'distance_km': float(edge_attr[0]),  # Edge attr[0] is distance
                'wind_alignment': float(edge_attr[1]),  # Edge attr[1] is wind alignment
                'slope_angle': float(edge_attr[2]),  # Edge attr[2] is slope angle
                'edge_attr_3': float(edge_attr[3]),
                'edge_attr_4': float(edge_attr[4]),
                'edge_attr_5': float(edge_attr[5]),
                'edge_attr_6': float(edge_attr[6]),
            }

            # Add individual metal contributions
            metal_names = self.base_config.metals
            for j, metal in enumerate(metal_names):
                data_row[f'contrib_{metal}'] = float(contrib_abs[j])

            xai_data.append(data_row)

        # Convert to DataFrame
        xai_df = pd.DataFrame(xai_data)

        logging.info(f"✓ Created XAI data table with {len(xai_df)} records")
        logging.info(f"  Columns: {list(xai_df.columns)}")

        # Save to CSV
        output_file = os.path.join(output_dir, "XAI_Real_Attention_Weights.csv")
        xai_df.to_csv(output_file, index=False, encoding='utf-8')
        logging.info(f"✓ Saved real attention weights to: {output_file}")

        # Print summary statistics
        logging.info("=" * 70)
        logging.info("Real Attention Weights Summary Statistics")
        logging.info("=" * 70)
        logging.info(f"Total links: {len(xai_df)}")
        logging.info(f"Attention weight - Mean: {xai_df['attention_weight'].mean():.4f}")
        logging.info(f"Attention weight - Std: {xai_df['attention_weight'].std():.4f}")
        logging.info(f"Attention weight - Min: {xai_df['attention_weight'].min():.4f}")
        logging.info(f"Attention weight - Max: {xai_df['attention_weight'].max():.4f}")
        logging.info(f"Attention weight - Median: {xai_df['attention_weight'].median():.4f}")

        logging.info("\nBy Source Type:")
        for source_type in xai_df['source_type'].unique():
            subset = xai_df[xai_df['source_type'] == source_type]
            logging.info(f"  {source_type}:")
            logging.info(f"    Count: {len(subset)}")
            logging.info(f"    Avg attention: {subset['attention_weight'].mean():.4f}")
            logging.info(f"    Avg distance: {subset['distance_km'].mean():.2f} km")

        logging.info("=" * 70)
        logging.info("✅ Real attention weights extraction complete!")
        logging.info("=" * 70)

