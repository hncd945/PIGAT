#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Parameter sensitivity analysis module for systematic hyperparameter sweeps.
Performs real training experiments to evaluate model robustness to parameter variations.

参数敏感性分析模块，用于系统性超参数扫描。
执行真实训练实验以评估模型对参数变化的鲁棒性。

Author: Wenhao Wang
"""
from __future__ import annotations

import os
import time
import copy
import logging
from typing import Dict, List, Any

import numpy as np
from omegaconf import OmegaConf

from .data_utils import PhysGATDataset
from .ensemble import EnsembleTrainer


def _clone_cfg(cfg) -> Any:
    """Deep copy omegaconf config as a new instance."""
    return OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _single_run(cfg, data_dir: str) -> Dict[str, Any]:
    """Run a single real training job and return summary metrics.

    We train with ensemble size possibly overridden to 1 for efficiency; epochs
    are taken from cfg.training.epochs (caller can override when constructing cfg).
    Performance metric for sensitivity is defined as 1 - final cosine loss
    (higher is better, in [0, 1]).
    """
    dataset = PhysGATDataset(root=data_dir, config=cfg)
    trainer = EnsembleTrainer(cfg, dataset)

    # Force ensemble size = 1 for sensitivity efficiency
    cfg.ensemble.n_ensemble = 1
    preds_df = trainer.train_ensemble()

    # representative history stores detailed loss tracks
    hist = {} if trainer.representative_history is None else trainer.representative_history.get('training_history', {})
    # Use reconstruction_loss instead of cosine_loss (updated naming)
    reconstruction_loss_track: List[float] = hist.get('reconstruction_loss', [])
    perf = 0.0
    if reconstruction_loss_track:
        final_recon = float(reconstruction_loss_track[-1])
        perf = float(max(0.0, 1.0 - final_recon))
    else:
        # Fallback: try legacy cosine_loss naming for backward compatibility
        cosine_loss_track: List[float] = hist.get('cosine_loss', [])
        if cosine_loss_track:
            final_cos = float(cosine_loss_track[-1])
            perf = float(max(0.0, 1.0 - final_cos))

    result = {
        'performance': perf,
        'final_losses': {
            'chemistry': float(hist.get('chemical_loss', [-1])[-1]) if hist.get('chemical_loss') else None,
            'distance': float(hist.get('distance_loss', [-1])[-1]) if hist.get('distance_loss') else None,
            'strength': float(hist.get('strength_loss', [-1])[-1]) if hist.get('strength_loss') else None,
            'cosine': float(reconstruction_loss_track[-1]) if reconstruction_loss_track else (float(cosine_loss_track[-1]) if cosine_loss_track else None),
            'total': float(hist.get('total_loss', [-1])[-1]) if hist.get('total_loss') else None,
        }
    }
    return result


def _default_value_grid(cfg) -> Dict[str, List[Any]]:
    """Return default 7-point grids for 4 key parameters."""
    current_thresh = getattr(cfg.graph, 'chem_sim_absolute_threshold', 0.7)
    current_k = getattr(cfg.graph, 'k_neighbors_s', 4)
    current_lr = getattr(cfg.training, 'learning_rate', 0.001)
    current_hc = getattr(cfg.model, 'hidden_channels', 128)

    return {
        'chem_sim_absolute_threshold': [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        'k_neighbors_s': [2, 4, 6, 8, 10, 12, 14],
        'learning_rate': [1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1],
        'hidden_channels': [32, 64, 128, 256, 512, 1024, 2048],
    }


def run_parameter_sensitivity(cfg, data_dir: str, output_dir: str) -> Dict[str, Any]:
    """Run real parameter sensitivity sweeps and return structured results.

    The function varies one parameter at a time while keeping others at default
    values. Each trial triggers a real training run. Results are stored to
    output_dir/sensitivity and also returned as a dict consumable by
    VisualizationSuite.figure15_parameter_sensitivity_analysis().
    """
    t0 = time.time()
    sens_out_dir = os.path.join(output_dir, 'sensitivity')
    _ensure_dir(sens_out_dir)

    # Configure epochs per trial for sensitivity (shorter but real runs)
    epochs_per_trial = int(getattr(cfg, 'sensitivity', {}).get('epochs_per_trial', 60))

    grids = _default_value_grid(cfg)
    results_for_viz: Dict[str, Dict[str, Any]] = {}

    for param, values in grids.items():
        logging.info(f"[Sensitivity] Sweeping parameter: {param} over {values}")
        performances: List[float] = []
        records: List[Dict[str, Any]] = []

        for v in values:
            cfg_trial = _clone_cfg(cfg)
            # Set trial epochs and ensemble size 1
            cfg_trial.training.epochs = epochs_per_trial
            cfg_trial.ensemble.n_ensemble = 1

            # Apply parameter value
            if param == 'chem_sim_absolute_threshold':
                cfg_trial.graph.chem_sim_absolute_threshold = float(v)
            elif param == 'k_neighbors_s':
                cfg_trial.graph.k_neighbors_s = int(v)
            elif param == 'learning_rate':
                cfg_trial.training.learning_rate = float(v)
            elif param == 'hidden_channels':
                cfg_trial.model.hidden_channels = int(v)
            else:
                logging.warning(f"Unknown parameter in sensitivity sweep: {param}")

            # Run a single training job
            try:
                trial_res = _single_run(cfg_trial, cfg_trial.data_dir)
            except Exception as e:
                logging.error(f"Sensitivity trial failed for {param}={v}: {e}")
                trial_res = {'performance': float('nan'), 'final_losses': {}}

            performances.append(float(trial_res['performance']))
            rec = {'param': param, 'value': v, 'performance': trial_res['performance']}
            rec.update({f"final_{k}": trial_res['final_losses'].get(k) for k in ['chemistry','distance','strength','cosine','total']})
            records.append(rec)

        # Save csv for this parameter
        import csv
        csv_path = os.path.join(sens_out_dir, f"sensitivity_{param}.csv")
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
            writer.writeheader()
            writer.writerows(records)
        logging.info(f"[Sensitivity] Saved results for {param} to {csv_path}")

        # Prepare structure for visualization
        xlabel_map = {
            'chem_sim_absolute_threshold': 'Chemical Similarity Threshold',
            'k_neighbors_s': 'Number of Neighbors (k)',
            'learning_rate': 'Learning Rate',
            'hidden_channels': 'Hidden Channels',
        }
        current_val = (
            getattr(cfg.graph, 'chem_sim_absolute_threshold', None) if param == 'chem_sim_absolute_threshold' else
            getattr(cfg.graph, 'k_neighbors_s', None) if param == 'k_neighbors_s' else
            getattr(cfg.training, 'learning_rate', None) if param == 'learning_rate' else
            getattr(cfg.model, 'hidden_channels', None) if param == 'hidden_channels' else None
        )

        results_for_viz[xlabel_map[param]] = {
            'values': values,
            'performance': performances,
            'current': current_val,
            'xlabel': xlabel_map[param]
        }

    elapsed = time.time() - t0
    logging.info(f"[Sensitivity] Completed all sweeps in {elapsed/60.0:.1f} minutes")

    # Return in the format expected by VisualizationSuite
    return results_for_viz

