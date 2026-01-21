#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Configuration validator module for validating and auto-fixing YAML configuration files.
Ensures all required parameters are present and within valid ranges before model execution.

配置验证模块，用于验证和自动修复YAML配置文件。
确保所有必需参数存在且在有效范围内后再执行模型。

Author: Wenhao Wang
"""

import os
import logging
from typing import Dict, Any, List, Tuple
import yaml
from omegaconf import DictConfig

class ConfigValidator:

    def __init__(self):
        self.validation_errors = []
        self.validation_warnings = []

    def validate_config(self, config: DictConfig) -> Tuple[bool, List[str], List[str]]:

        self.validation_errors = []
        self.validation_warnings = []

        self._validate_basic_settings(config)

        self._validate_data_settings(config)

        self._validate_model_settings(config)

        self._validate_training_settings(config)

        self._validate_graph_settings(config)

        self._validate_pmf_settings(config)

        self._validate_ensemble_settings(config)

        # NEW: Validate extended configuration sections (v2.0)
        self._validate_convergence_study_settings(config)

        self._validate_edge_construction_settings(config)

        self._validate_loss_weights_settings(config)

        self._validate_visualization_settings(config)

        is_valid = len(self.validation_errors) == 0
        return is_valid, self.validation_errors, self.validation_warnings

    def _validate_basic_settings(self, config: DictConfig):
        if not hasattr(config, 'seed') or not isinstance(config.seed, int):
            self.validation_errors.append("seed must be an integer")
        elif config.seed < 0:
            self.validation_warnings.append("recommend using non-negative number as random seed")

        if not hasattr(config, 'device'):
            self.validation_errors.append("missing device setting")
        elif config.device not in ['cuda', 'cpu']:
            self.validation_errors.append("device must be 'cuda' or 'cpu'")

        if hasattr(config, 'environment') and hasattr(config.environment, 'prevailing_wind_direction'):
            pwd = config.environment.prevailing_wind_direction
            if not isinstance(pwd, (int, float)):
                self.validation_errors.append("environment.prevailing_wind_direction must be numeric")
            elif not (0 <= float(pwd) < 360):
                self.validation_warnings.append("environment.prevailing_wind_direction should be in [0, 360)")

    def _validate_data_settings(self, config: DictConfig):
        # Check for data_dir in both old location (top-level) and new location (data section)
        data_dir_found = False
        if hasattr(config, 'data_dir'):
            data_dir_found = True
            if not isinstance(config.data_dir, str):
                self.validation_errors.append("data_dir must be a string")
        elif hasattr(config, 'data') and hasattr(config.data, 'data_dir'):
            data_dir_found = True
            if not isinstance(config.data.data_dir, str):
                self.validation_errors.append("data.data_dir must be a string")

        if not data_dir_found:
            self.validation_errors.append("missing data_dir setting (check top-level or data section)")

        if hasattr(config, 'dem_path') and config.dem_path:
            if not os.path.exists(config.dem_path):
                self.validation_warnings.append(f"DEM file does not exist: {config.dem_path}")
        elif hasattr(config, 'data') and hasattr(config.data, 'dem_path') and config.data.dem_path:
            if not os.path.exists(config.data.dem_path):
                self.validation_warnings.append(f"DEM file does not exist: {config.data.dem_path}")

        # Check for metals in both old location (top-level) and new location (data section)
        metals_found = False
        metals_list = None
        if hasattr(config, 'metals'):
            metals_found = True
            metals_list = config.metals
        elif hasattr(config, 'data') and hasattr(config.data, 'metals'):
            metals_found = True
            metals_list = config.data.metals

        if not metals_found:
            self.validation_errors.append("missing metals setting (check top-level or data section)")
        elif metals_list is None:
            self.validation_errors.append("metals cannot be empty")
        elif not hasattr(metals_list, '__iter__'):
            self.validation_errors.append("metals must be a list")
        elif len(metals_list) == 0:
            self.validation_errors.append("metals list cannot be empty")
        else:
            valid_metals = ['Cr', 'Ni', 'Cu', 'Zn', 'As', 'Cd', 'Pb', 'Hg']
            for metal in metals_list:
                if metal not in valid_metals:
                    self.validation_warnings.append(f"unknown heavy metal: {metal}")

    def _validate_model_settings(self, config: DictConfig):
        if not hasattr(config, 'model'):
            self.validation_errors.append("missing model setting")
            return

        model_config = config.model

        if not hasattr(model_config, 'hidden_channels'):
            self.validation_errors.append("missing model.hidden_channels setting")
        elif not isinstance(model_config.hidden_channels, int) or model_config.hidden_channels <= 0:
            self.validation_errors.append("model.hidden_channels must be a positive integer")
        elif model_config.hidden_channels < 32:
            self.validation_warnings.append("hidden_channels is small, may affect model expressiveness")
        elif model_config.hidden_channels > 512:
            self.validation_warnings.append("hidden_channels is large, may cause overfitting")

        if not hasattr(model_config, 'out_channels'):
            self.validation_errors.append("missing model.out_channels setting")
        elif not isinstance(model_config.out_channels, int) or model_config.out_channels <= 0:
            self.validation_errors.append("model.out_channels must be a positive integer")

        if not hasattr(model_config, 'num_heads'):
            self.validation_errors.append("missing model.num_heads setting")
        elif not isinstance(model_config.num_heads, int) or model_config.num_heads <= 0:
            self.validation_errors.append("model.num_heads must be a positive integer")
        elif model_config.num_heads > 16:
            self.validation_warnings.append("num_heads too large may increase computational overhead")

        if not hasattr(model_config, 'dropout_rate'):
            self.validation_errors.append("missing model.dropout_rate setting")
        elif not isinstance(model_config.dropout_rate, (int, float)):
            self.validation_errors.append("model.dropout_rate must be numeric")
        elif not (0 <= model_config.dropout_rate <= 1):
            self.validation_errors.append("model.dropout_rate must be between 0-1")

    def _validate_training_settings(self, config: DictConfig):
        if not hasattr(config, 'training'):
            self.validation_errors.append("missing training setting")
            return

        training_config = config.training

        if not hasattr(training_config, 'learning_rate'):
            self.validation_errors.append("missing training.learning_rate setting")
        elif not isinstance(training_config.learning_rate, (int, float)):
            self.validation_errors.append("training.learning_rate must be numeric")
        elif training_config.learning_rate <= 0:
            self.validation_errors.append("training.learning_rate must be positive")
        elif training_config.learning_rate > 0.1:
            self.validation_warnings.append("learning_rate too large may cause training instability")
        elif training_config.learning_rate < 1e-6:
            self.validation_warnings.append("learning_rate too small may cause slow training")

        if not hasattr(training_config, 'epochs'):
            self.validation_errors.append("missing training.epochs setting")
        elif not isinstance(training_config.epochs, int) or training_config.epochs <= 0:
            self.validation_errors.append("training.epochs must be a positive integer")
        elif training_config.epochs < 50:
            self.validation_warnings.append("few epochs may cause insufficient training")
        elif training_config.epochs > 1000:
            self.validation_warnings.append("too many epochs may cause overfitting")

        if hasattr(training_config, 'weight_decay'):
            if not isinstance(training_config.weight_decay, (int, float)):
                self.validation_errors.append("training.weight_decay must be numeric")
            elif training_config.weight_decay < 0:
                self.validation_errors.append("training.weight_decay cannot be negative")

    def _validate_graph_settings(self, config: DictConfig):
        """Validate graph construction settings"""
        if not hasattr(config, 'graph'):
            self.validation_errors.append("Missing graph settings")
            return

        graph_config = config.graph

        # Distance decay
        if hasattr(graph_config, 'distance_decay_km'):
            dd = graph_config.distance_decay_km
            if not isinstance(dd, (int, float)) or dd <= 0:
                self.validation_errors.append("graph.distance_decay_km must be a positive number")
            elif dd > 50:
                self.validation_warnings.append(f"graph.distance_decay_km is large ({dd}km), may over-penalize distance")

        # Terrain settings
        if hasattr(graph_config, 'terrain') and hasattr(graph_config.terrain, 'elevation_diff_scale_km'):
            es = graph_config.terrain.elevation_diff_scale_km
            if not isinstance(es, (int, float)) or es <= 0:
                self.validation_errors.append("graph.terrain.elevation_diff_scale_km must be a positive number")

        # Irrigation path params
        if hasattr(graph_config, 'irrigation'):
            irr = graph_config.irrigation
            if hasattr(irr, 'direct_distance_threshold_km'):
                th = irr.direct_distance_threshold_km
                if not isinstance(th, (int, float)) or th <= 0:
                    self.validation_errors.append("graph.irrigation.direct_distance_threshold_km must be positive")
            if hasattr(irr, 'curve_multiplier'):
                cm = irr.curve_multiplier
                if not isinstance(cm, (int, float)) or cm < 1:
                    self.validation_warnings.append("graph.irrigation.curve_multiplier should be >= 1")

        # Check k_neighbors_r
        if not hasattr(graph_config, 'k_neighbors_r'):
            self.validation_errors.append("Missing graph.k_neighbors_r setting")
        elif not isinstance(graph_config.k_neighbors_r, int) or graph_config.k_neighbors_r <= 0:
            self.validation_errors.append("graph.k_neighbors_r must be a positive integer")

        # Check distance thresholds
        if hasattr(graph_config, 'max_distance_km'):
            distance_config = graph_config.max_distance_km
            required_source_types = ['atmosphere', 'irrigation', 'fertilizer', 'organic']

            for source_type in required_source_types:
                if hasattr(distance_config, source_type):
                    dist = getattr(distance_config, source_type)
                    if not isinstance(dist, (int, float)) or dist <= 0:
                        self.validation_errors.append(f"graph.max_distance_km.{source_type} must be a positive number")
                    elif dist > 50:
                        self.validation_warnings.append(f"Distance threshold for {source_type} sources is very large ({dist}km)")

        # Check candidate selection settings
        if hasattr(graph_config, 'candidate_selection'):
            candidate_config = graph_config.candidate_selection

            # Check max_candidates_per_type
            if hasattr(candidate_config, 'max_candidates_per_type'):
                max_candidates = candidate_config.max_candidates_per_type
                if not isinstance(max_candidates, int) or max_candidates <= 0:
                    self.validation_errors.append("graph.candidate_selection.max_candidates_per_type must be a positive integer")
                elif max_candidates > 16:
                    self.validation_warnings.append(f"High number of candidates per type ({max_candidates}) may reduce filtering effectiveness")

            # Check max_total_connections_per_receptor
            if hasattr(candidate_config, 'max_total_connections_per_receptor'):
                max_total = candidate_config.max_total_connections_per_receptor
                if not isinstance(max_total, int) or max_total <= 0:
                    self.validation_errors.append("graph.candidate_selection.max_total_connections_per_receptor must be a positive integer")
                elif max_total > 20:
                    self.validation_warnings.append(f"High maximum connections per receptor ({max_total}) may reduce sparsity")

            # Check scoring weights (per-source-type structure)
            if hasattr(candidate_config, 'scoring_weights'):
                sw = candidate_config.scoring_weights
                required_weights = ['distance_score', 'chemical_score', 'strength_score', 'wind_score']

                def _check_one_group(group_obj, group_name: str):
                    if group_obj is None:
                        self.validation_warnings.append(f"Missing scoring weight group: {group_name}")
                        return
                    wsum = 0.0
                    present = False
                    for key in required_weights:
                        if hasattr(group_obj, key):
                            val = getattr(group_obj, key)
                            present = True
                            if not isinstance(val, (int, float)) or val < 0:
                                self.validation_errors.append(
                                    f"graph.candidate_selection.scoring_weights.{group_name}.{key} must be non-negative number")
                            else:
                                wsum += float(val)
                        else:
                            # Not all keys must exist; treat missing as 0 and warn
                            self.validation_warnings.append(
                                f"graph.candidate_selection.scoring_weights.{group_name}.{key} missing; treated as 0")
                    if wsum <= 0.0:
                        self.validation_errors.append(
                            f"graph.candidate_selection.scoring_weights.{group_name} has zero total weight; cannot normalize")
                    elif abs(wsum - 1.0) > 0.01:
                        self.validation_warnings.append(
                            f"scoring_weights.{group_name} sum = {wsum:.3f}; will be normalized per type")

                # Known groups
                for g in ['atmosphere', 'irrigation', 'fertilizer', 'organic', 'default']:
                    if hasattr(sw, g):
                        _check_one_group(getattr(sw, g), g)

                # Zero-candidate exception groups (optional)
                if hasattr(candidate_config, 'zero_candidate_exception'):
                    zce = candidate_config.zero_candidate_exception
                    for g in ['atmosphere_exception', 'irrigation_exception', 'fertilizer_exception', 'organic_exception']:
                        if hasattr(zce, g):
                            _check_one_group(getattr(zce, g), g)

        # Check chemical similarity threshold
        if hasattr(graph_config, 'chem_sim_absolute_threshold'):
            if not isinstance(graph_config.chem_sim_absolute_threshold, (int, float)):
                self.validation_errors.append("graph.chem_sim_absolute_threshold must be numeric")
            elif not (0 <= graph_config.chem_sim_absolute_threshold <= 1):
                self.validation_errors.append("graph.chem_sim_absolute_threshold must be between 0 and 1")

    def _validate_pmf_settings(self, config: DictConfig):
        if not hasattr(config, 'pmf'):
            self.validation_warnings.append("missing pmf setting, will use default values")
            return

        pmf_config = config.pmf

        if hasattr(pmf_config, 'n_components'):
            if not isinstance(pmf_config.n_components, int) or pmf_config.n_components <= 0:
                self.validation_errors.append("pmf.n_components must be a positive integer")
            elif pmf_config.n_components > 10:
                self.validation_warnings.append("too many PMF components may cause overfitting")

    def _validate_ensemble_settings(self, config: DictConfig):
        if not hasattr(config, 'ensemble'):
            self.validation_warnings.append("Missing ensemble settings; using defaults")
            return

        ensemble_config = config.ensemble

        #n_ensemble
        if hasattr(ensemble_config, 'n_ensemble'):
            if not isinstance(ensemble_config.n_ensemble, int) or ensemble_config.n_ensemble <= 0:
                self.validation_errors.append("ensemble.n_ensemble must be a positive integer")
            elif ensemble_config.n_ensemble < 3:
                self.validation_warnings.append("Small ensemble size may affect prediction stability")
            elif ensemble_config.n_ensemble > 20:
                self.validation_warnings.append("Large ensemble size may cause excessive compute cost")

    def _validate_convergence_study_settings(self, config: DictConfig):
        """Validate convergence_study settings (NEW in v2.0)"""
        if not hasattr(config, 'convergence_study'):
            self.validation_warnings.append("Missing convergence_study settings; using defaults")
            return

        cs = config.convergence_study

        # Check enabled flag
        if hasattr(cs, 'enabled'):
            if not isinstance(cs.enabled, bool):
                self.validation_errors.append("convergence_study.enabled must be boolean")

        # Check epochs
        if hasattr(cs, 'epochs'):
            if not isinstance(cs.epochs, int) or cs.epochs <= 0:
                self.validation_errors.append("convergence_study.epochs must be a positive integer")
            elif cs.epochs < 100:
                self.validation_warnings.append("convergence_study.epochs < 100 may be insufficient for convergence study")

        # Check n_ensemble
        if hasattr(cs, 'n_ensemble'):
            if not isinstance(cs.n_ensemble, int) or cs.n_ensemble <= 0:
                self.validation_errors.append("convergence_study.n_ensemble must be a positive integer")
            elif cs.n_ensemble < 3:
                self.validation_warnings.append("convergence_study.n_ensemble < 3 may not provide reliable uncertainty estimates")

    def _validate_edge_construction_settings(self, config: DictConfig):
        """Validate edge_construction settings (NEW in v2.0)"""
        if not hasattr(config, 'edge_construction'):
            self.validation_warnings.append("Missing edge_construction settings; using defaults")
            return

        ec = config.edge_construction

        # Validate distance_thresholds
        if hasattr(ec, 'distance_thresholds'):
            dt = ec.distance_thresholds
            required_types = ['atmosphere', 'irrigation', 'fertilizer', 'organic']

            for source_type in required_types:
                if hasattr(dt, source_type):
                    threshold = getattr(dt, source_type)
                    if not isinstance(threshold, (int, float)) or threshold <= 0:
                        self.validation_errors.append(
                            f"edge_construction.distance_thresholds.{source_type} must be a positive number")
                    elif threshold > 100:
                        self.validation_warnings.append(
                            f"edge_construction.distance_thresholds.{source_type} is very large ({threshold}km)")
                else:
                    self.validation_warnings.append(
                        f"Missing edge_construction.distance_thresholds.{source_type}")
        else:
            self.validation_errors.append("Missing edge_construction.distance_thresholds")

        # Validate max_candidates_per_type
        if hasattr(ec, 'max_candidates_per_type'):
            mc = ec.max_candidates_per_type
            for source_type in ['atmosphere', 'irrigation', 'fertilizer', 'organic']:
                if hasattr(mc, source_type):
                    max_count = getattr(mc, source_type)
                    if not isinstance(max_count, int) or max_count <= 0:
                        self.validation_errors.append(
                            f"edge_construction.max_candidates_per_type.{source_type} must be a positive integer")

        # Validate global constraints
        if hasattr(ec, 'global_min_candidates'):
            if not isinstance(ec.global_min_candidates, int) or ec.global_min_candidates <= 0:
                self.validation_errors.append("edge_construction.global_min_candidates must be a positive integer")

        if hasattr(ec, 'global_max_candidates'):
            if not isinstance(ec.global_max_candidates, int) or ec.global_max_candidates <= 0:
                self.validation_errors.append("edge_construction.global_max_candidates must be a positive integer")
            elif hasattr(ec, 'global_min_candidates') and ec.global_max_candidates < ec.global_min_candidates:
                self.validation_errors.append(
                    "edge_construction.global_max_candidates must be >= global_min_candidates")

        # Validate scoring_weights
        if hasattr(ec, 'scoring_weights'):
            sw = ec.scoring_weights
            for source_type in ['atmosphere', 'irrigation', 'fertilizer', 'organic']:
                if hasattr(sw, source_type):
                    type_weights = getattr(sw, source_type)
                    weight_sum = 0.0
                    for key in ['distance_score', 'wind_score', 'chemical_score', 'strength_score', 'hydro_score']:
                        if hasattr(type_weights, key):
                            val = getattr(type_weights, key)
                            if not isinstance(val, (int, float)) or val < 0:
                                self.validation_errors.append(
                                    f"edge_construction.scoring_weights.{source_type}.{key} must be non-negative")
                            else:
                                weight_sum += float(val)

                    if weight_sum <= 0:
                        self.validation_errors.append(
                            f"edge_construction.scoring_weights.{source_type} has zero total weight")
                    elif abs(weight_sum - 1.0) > 0.01:
                        self.validation_warnings.append(
                            f"edge_construction.scoring_weights.{source_type} sum = {weight_sum:.3f}; should be ~1.0")

    def _validate_loss_weights_settings(self, config: DictConfig):
        """Validate loss_weights settings (NEW in v2.0)"""
        if not hasattr(config, 'loss_weights'):
            self.validation_warnings.append("Missing loss_weights settings; using defaults")
            return

        lw = config.loss_weights

        # Validate scaling factors
        for scale_name in ['chemistry_scale', 'distance_scale', 'structure_scale', 'reconstruction_scale']:
            if hasattr(lw, scale_name):
                scale_val = getattr(lw, scale_name)
                if not isinstance(scale_val, (int, float)) or scale_val <= 0:
                    self.validation_errors.append(f"loss_weights.{scale_name} must be a positive number")
                elif scale_val > 10:
                    self.validation_warnings.append(
                        f"loss_weights.{scale_name} is very large ({scale_val}); may cause loss imbalance")
            else:
                self.validation_warnings.append(f"Missing loss_weights.{scale_name}")

        # Validate adaptive_weights
        if hasattr(lw, 'adaptive_weights'):
            aw = lw.adaptive_weights

            if hasattr(aw, 'enabled'):
                if not isinstance(aw.enabled, bool):
                    self.validation_errors.append("loss_weights.adaptive_weights.enabled must be boolean")

            if hasattr(aw, 'clamp_range'):
                clamp_range = aw.clamp_range
                # Handle both list and ListConfig from OmegaConf
                try:
                    clamp_list = list(clamp_range)
                    if len(clamp_list) != 2:
                        self.validation_errors.append("loss_weights.adaptive_weights.clamp_range must have exactly 2 values [min, max]")
                    else:
                        min_val, max_val = clamp_list
                        if not isinstance(min_val, (int, float)) or not isinstance(max_val, (int, float)):
                            self.validation_errors.append("loss_weights.adaptive_weights.clamp_range values must be numeric")
                        elif min_val >= max_val:
                            self.validation_errors.append("loss_weights.adaptive_weights.clamp_range: min must be < max")
                        elif min_val < 0 or max_val > 1:
                            self.validation_warnings.append(
                                "loss_weights.adaptive_weights.clamp_range should be within [0, 1]")
                except (TypeError, ValueError):
                    self.validation_errors.append("loss_weights.adaptive_weights.clamp_range must be a list [min, max]")

            if hasattr(aw, 'chemistry_min_sum'):
                cms = aw.chemistry_min_sum
                if not isinstance(cms, (int, float)) or cms < 0 or cms > 1:
                    self.validation_errors.append(
                        "loss_weights.adaptive_weights.chemistry_min_sum must be in [0, 1]")

    def _validate_visualization_settings(self, config: DictConfig):
        """Validate visualization settings (NEW in v2.0)"""
        if not hasattr(config, 'visualization'):
            self.validation_warnings.append("Missing visualization settings; using defaults")
            return

        viz = config.visualization

        # Validate source_type_mapping
        if hasattr(viz, 'source_type_mapping'):
            stm = viz.source_type_mapping
            required_mappings = ['atmospheric', 'industrial', 'pesticide', 'manure']
            for old_name in required_mappings:
                if not hasattr(stm, old_name):
                    self.validation_warnings.append(
                        f"Missing visualization.source_type_mapping.{old_name}")
        else:
            self.validation_warnings.append("Missing visualization.source_type_mapping")

        # Validate source_type_order
        if hasattr(viz, 'source_type_order'):
            sto = viz.source_type_order
            # Handle both list and ListConfig from OmegaConf
            try:
                sto_list = list(sto)
                if len(sto_list) != 4:
                    self.validation_warnings.append(
                        f"visualization.source_type_order has {len(sto_list)} types; expected 4")
            except (TypeError, ValueError):
                self.validation_errors.append("visualization.source_type_order must be a list")

        # Validate source_colors
        if hasattr(viz, 'source_colors'):
            sc = viz.source_colors
            for source_type in ['atmosphere', 'irrigation', 'fertilizer', 'organic']:
                if hasattr(sc, source_type):
                    color = getattr(sc, source_type)
                    if not isinstance(color, str):
                        self.validation_errors.append(
                            f"visualization.source_colors.{source_type} must be a string")
                    elif not color.startswith('#'):
                        self.validation_warnings.append(
                            f"visualization.source_colors.{source_type} should be hex color (e.g., #FF0000)")

def validate_and_fix_config(config: DictConfig) -> Tuple[DictConfig, bool, List[str], List[str]]:

    validator = ConfigValidator()
    is_valid, errors, warnings = validator.validate_config(config)

    if errors:
        logging.error("Configuration validation failed:")
        for error in errors:
            logging.error(f"  - {error}")

    if warnings:
        logging.warning("Configuration validation warnings:")
        for warning in warnings:
            logging.warning(f"  - {warning}")

    if is_valid:
        logging.info("Configuration validation passed")

    return config, is_valid, errors, warnings
