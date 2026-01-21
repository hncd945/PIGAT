#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Main entry point for the PhysGAT soil pollution source apportionment system.
Orchestrates data loading, model training, ensemble learning, and visualization generation.

PhysGAT土壤污染溯源系统主入口。
负责协调数据加载、模型训练、集成学习和可视化图表生成的完整工作流程。

Author: Wenhao Wang
"""

try:
    from src.proj_fix import apply_proj_fix
    apply_proj_fix()
except ImportError:
    import warnings
    warnings.filterwarnings('ignore', category=UserWarning, module='rasterio')
    warnings.filterwarnings('ignore', message='.*PROJ.*DATABASE.LAYOUT.VERSION.*')

import hydra
from omegaconf import DictConfig, OmegaConf
import logging
import os
import pandas as pd

from src.data_utils import PhysGATDataset
from src.ensemble import EnsembleTrainer
from src.visualization import VisualizationSuite
from src.analysis import run_full_analysis, analyze_explanation, run_gnnexplainer_analysis
from src.utils import set_seed
from src.config_validator import validate_and_fix_config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

@hydra.main(config_path="configs", config_name="default", version_base=None)
def main(cfg: DictConfig) -> None:
    logging.info("=== Configuration validation ===")
    cfg, is_valid, errors, warnings = validate_and_fix_config(cfg)

    if not is_valid:
        logging.error("Configuration validation failed. Exiting.")
        for error in errors:
            logging.error(f"Config error: {error}")
        return

    if warnings:
        logging.warning("Configuration validation warnings:")
        for warning in warnings:
            logging.warning(f"  - {warning}")

    set_seed(cfg.seed)

    from hydra.core.hydra_config import HydraConfig
    output_dir = HydraConfig.get().runtime.output_dir
    os.makedirs(output_dir, exist_ok=True)

    log_conf_path = os.path.join(output_dir, "logs")
    os.makedirs(log_conf_path, exist_ok=True)
    OmegaConf.update(cfg, "output_dir", output_dir, merge=True)

    # Convergence study mode (optional, controlled by config)
    # This replaces the previous forced override approach
    if hasattr(cfg, 'convergence_study') and cfg.convergence_study.get('enabled', False):
        logging.info("Convergence study mode enabled")
        try:
            from omegaconf import OmegaConf as _OC
            _OC.update(cfg, "training.epochs", cfg.convergence_study.epochs, merge=True)
            _OC.update(cfg, "ensemble.n_ensemble", cfg.convergence_study.n_ensemble, merge=True)
            logging.info(f"  - Training epochs: {cfg.training.epochs}")
            logging.info(f"  - Ensemble size: {cfg.ensemble.n_ensemble}")
        except Exception as _e:
            logging.warning(f"Failed to apply convergence study settings: {_e}")
    else:
        logging.info("Using standard training mode")
        logging.info(f"  - Training epochs: {cfg.training.epochs}")
        logging.info(f"  - Ensemble size: {cfg.ensemble.n_ensemble}")

    OmegaConf.update(cfg, "log_dir", log_conf_path, merge=True)

    logging.info("=" * 70)
    logging.info("PhysGAT POLLUTION SOURCE TRACING SYSTEM")
    logging.info("Physics-informed Graph Attention Network")
    logging.info("=" * 70)
    logging.info(f"Output Directory: {output_dir}")
    logging.info(f"Training Epochs: {cfg.training.epochs}")
    logging.info(f"Ensemble Size: {cfg.ensemble.n_ensemble}")
    logging.info(f"Device: {cfg.device}")
    logging.info("=" * 70)

    if logging.getLogger().level <= logging.DEBUG:
        logging.debug(f"Full Config contents:\n{OmegaConf.to_yaml(cfg)}")

    # Handle data_dir path (Hydra changes working directory to run_outputs/)
    # CRITICAL: Use __file__ to get the directory where main.py is located
    # This ensures paths work regardless of where the script is run from
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Get project root (parent of physgat_project directory)
    project_root = os.path.dirname(script_dir)

    data_path = cfg.data_dir
    if not os.path.isabs(data_path):
        # Resolve relative path from main.py's directory (physgat_project/)
        data_path = os.path.normpath(os.path.join(script_dir, data_path))

    # Verify data directory exists
    if not os.path.exists(data_path):
        logging.error(f"Data directory not found: {data_path}")
        logging.error(f"Script directory: {script_dir}")
        logging.error(f"Config data_dir: {cfg.data_dir}")
        logging.error(f"Expected path: {os.path.join(script_dir, cfg.data_dir)}")
        raise FileNotFoundError(f"Data directory not found: {data_path}")

    # Also fix dem_path if it's relative
    if hasattr(cfg, 'dem_path') and cfg.dem_path:
        dem_path = cfg.dem_path
        if not os.path.isabs(dem_path):
            # Resolve relative to project root (not physgat_project directory)
            dem_path = os.path.normpath(os.path.join(project_root, dem_path))
            OmegaConf.update(cfg, "dem_path", dem_path, merge=True)
            logging.info(f"Resolved DEM path: {dem_path}")

    logging.info(f"Data directory: {data_path}")
    dataset = PhysGATDataset(root=data_path, config=cfg)
    data = dataset[0]
    logging.info(f"Data loading complete. Graph contains {data.x_receptor.shape[0]} receptors and {data.x_source.shape[0]} sources.")

    ensemble_trainer = EnsembleTrainer(cfg, dataset)
    ensemble_predictions_df = ensemble_trainer.train_ensemble()

    logging.info("=== Independent Training Mode ===")
    logging.info("All models trained independently without consensus mechanisms")
    logging.info("============================================")

    logging.info("Starting post-processing analysis...")
    analysis_results = run_full_analysis(ensemble_predictions_df, dataset, cfg)

    # Note: Parameter sensitivity analysis has been moved to run_sensitivity_analysis.py
    # Run it separately to avoid extending main training workflow runtime
    # Usage: python run_sensitivity_analysis.py

    if ensemble_trainer.representative_trainer and analysis_results.get('receptors_with_nemerow') is not None:
        receptors_analyzed = analysis_results['receptors_with_nemerow']
        if not receptors_analyzed.empty:
            top_polluted_idx = receptors_analyzed.nlargest(1, 'nemerow_index').index[0]
            analyze_explanation(ensemble_trainer.representative_trainer, data, top_polluted_idx)
            run_gnnexplainer_analysis(ensemble_trainer.representative_trainer, data, top_polluted_idx)

    logging.info("Generating visualizations...")
    model_data_for_viz = {
        'receptors_df': dataset.receptors_df,
        'sources_df': dataset.sources_df,
        'training_history': ensemble_trainer.representative_history.get('training_history', {}) if ensemble_trainer.representative_history else {},
        'ensemble_training_history': ensemble_trainer.training_history if hasattr(ensemble_trainer, 'training_history') else {},
        'pmf_diagnostics': dataset.pmf_diagnostics,
        'weight_history': getattr(ensemble_trainer.representative_trainer, 'weight_history', {}) if ensemble_trainer.representative_trainer else {},
        'loss_history': getattr(ensemble_trainer.representative_trainer, 'loss_history', {}) if ensemble_trainer.representative_trainer else {}
    }
    viz_suite = VisualizationSuite(cfg, model_data_for_viz, analysis_results, output_dir)
    viz_suite.generate_all_figures()

    # Compute and persist manuscript metrics for reproducible reporting
    try:
        from src.analysis import compute_paper_metrics
        metrics = compute_paper_metrics(analysis_results, dataset, cfg)
        import json
        metrics_path = os.path.join(output_dir, "paper_metrics.json")
        with open(metrics_path, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)
        logging.info(f"Saved manuscript metrics to: {metrics_path}")
    except Exception as e:
        logging.error(f"Failed to compute/save paper metrics: {e}")

    logging.info("Saving key analysis results...")
    try:
        report_path = os.path.join(output_dir, "Final_Ensemble_Report.xlsx")
        with pd.ExcelWriter(report_path) as writer:
            top5_clean = analysis_results['top5_contribution_report'].copy()
            columns_to_remove = ['is_combined_source']
            for col in columns_to_remove:
                if col in top5_clean.columns:
                    top5_clean = top5_clean.drop(columns=[col])

            top5_clean.to_excel(writer, sheet_name='All_Contributions', index=False)
            analysis_results['contribution_pivot_summary'].to_excel(writer, sheet_name='Contribution_Summary_Pivot', index=False)

            if 'fallback_candidates' in analysis_results and not analysis_results['fallback_candidates'].empty:
                analysis_results['fallback_candidates'].to_excel(writer, sheet_name='Fallback_Candidates', index=False)
                logging.info("Added fallback candidate list (report-only; no effect on training or normalization)")

            if 'primary_secondary_pollutants' in analysis_results:
                analysis_results['primary_secondary_pollutants'].to_excel(writer, sheet_name='Primary_Secondary_Pollutants', index=False)
                logging.info("Added primary/secondary pollutant analysis sheet")

            if 'receptors_with_nemerow' in analysis_results:
                receptors_clean = analysis_results['receptors_with_nemerow'].copy()
                # Remove all background-related columns due to unified background approach
                bg_columns_to_remove = [col for col in receptors_clean.columns if
                                       'bg_' in col.lower() or 'background' in col.lower() or
                                       'combined' in col.lower() or 'merge' in col.lower()]
                for col in bg_columns_to_remove:
                    receptors_clean = receptors_clean.drop(columns=[col])

                receptors_clean.to_excel(writer, sheet_name='Filtered_Receptors_Analysis', index=False)
                logging.info("Added filtered receptor analysis sheet (post-Nemerow filtering)")

        logging.info(f"Analysis report saved to: {report_path}")
    except Exception as e:
        logging.error(f"Failed to save analysis report: {e}")

    logging.info("=" * 70)
    logging.info("PhysGAT WORKFLOW COMPLETED SUCCESSFULLY!")
    logging.info(f"Results saved in: {output_dir}")
    logging.info(f"Final report: {report_path}")
    logging.info("=" * 70)

if __name__ == "__main__":
    main()
