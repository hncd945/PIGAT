#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Professional visualization suite for generating publication-quality figures (Figures 1-16).
Creates source fingerprints, spatial distributions, training curves, heatmaps, and 3D terrain visualizations.

专业可视化套件，用于生成出版级质量的图表（图1-16）。
生成污染源指纹图、空间分布图、训练曲线、热力图和3D地形可视化。

Author: Wenhao Wang
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
import logging
from matplotlib.lines import Line2D
from matplotlib.ticker import FormatStrFormatter
from typing import Dict, Any
from matplotlib.colors import LightSource


# Try to import networkx, disable related features if failed
try:
    import networkx as nx
    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False

warnings.filterwarnings('ignore')

class VisualizationSuite:
    """
    Professional visualization module for generating all project report charts.
    """

    # Class-level constants for source type mapping and ordering
    # These are shared across multiple visualization methods (Figure 11, Figure 12, etc.)
    SOURCE_TYPE_MAPPING = {
        'atmospheric': 'atmosphere',  # atmospheric → atmosphere
        'industrial': 'irrigation',   # industrial → irrigation
        'pesticide': 'fertilizer'     # pesticide → fertilizer
    }

    SOURCE_TYPE_ORDER = ['atmosphere', 'irrigation', 'fertilizer', 'manure']

    def __init__(self, config: Dict[str, Any], model_data: Dict[str, Any],
                 analysis_results: Dict[str, pd.DataFrame], output_dir: str):
        """
        Initialize visualization suite.
        """
        self.config = config
        self.model_data = model_data
        self.analysis_results = analysis_results
        self.predictions_df = analysis_results.get('contribution_detailed_summary')
        self.receptors_df = model_data.get('receptors_df')
        self.sources_df = model_data.get('sources_df')
        self.output_dir = output_dir
        self.combination_details = analysis_results.get('source_combination_details', [])

        # Fix: Add metals attribute from config
        self.metals = config.get('metals', ['Cr', 'Ni', 'Cu', 'Zn', 'As', 'Cd', 'Pb'])

        # Fix: Add source colors mapping for Figure 1
        self.source_colors = {
            'atmosphere': '#E74C3C',    # Red
            'irrigation': '#3498DB',    # Blue
            'fertilizer': '#F39C12',    # Orange
            'manure': '#2ECC71'         # Green
        }

        os.makedirs(self.output_dir, exist_ok=True)

        # Set international journal standard chart styles with unified Times New Roman font
        try:
            plt.rcParams['font.family'] = 'Times New Roman'
        except Exception as e:
            logging.warning(f"Failed to set Times New Roman font, fallback to serif: {e}")
            plt.rcParams['font.family'] = 'serif'

        # International journal standard settings - enhanced resolution with unified font sizes
        plt.rcParams['axes.unicode_minus'] = False
        plt.rcParams['figure.dpi'] = 400
        plt.rcParams['savefig.dpi'] = 400
        plt.rcParams['font.size'] = 11
        plt.rcParams['axes.titlesize'] = 14
        plt.rcParams['axes.labelsize'] = 12
        plt.rcParams['xtick.labelsize'] = 11
        plt.rcParams['ytick.labelsize'] = 11
        plt.rcParams['legend.fontsize'] = 11
        plt.rcParams['figure.titlesize'] = 14
        plt.rcParams['lines.linewidth'] = 1.5
        plt.rcParams['axes.linewidth'] = 1.0

        # Unified chart styling parameters
        self.unified_legend_style = {
            'loc': 'upper right',
            'frameon': True,
            'fancybox': True,
            'shadow': True,
            'framealpha': 0.9,
            'facecolor': 'white',
            'edgecolor': 'black',
            'prop': {'family': 'Times New Roman', 'size': 11}
        }

        self.unified_border_style = {
            'linewidth': 1.0,
            'color': 'black'  # Use pure black border per user request
        }

    def _apply_unified_legend_style(self, ax, handles=None, labels=None, **kwargs):
        """Apply unified legend style to axes

        Args:
            ax: matplotlib axes object
            handles: list of legend handles (Line2D, Patch, etc.)
            labels: list of legend labels (optional if handles already have labels)
            **kwargs: additional legend parameters to override defaults

        Returns:
            matplotlib Legend object
        """
        legend_kwargs = self.unified_legend_style.copy()
        legend_kwargs.update(kwargs)

        if handles is not None and labels is not None:
            # Both handles and labels provided
            return ax.legend(handles, labels, **legend_kwargs)
        elif handles is not None:
            # Only handles provided (labels should be in handle objects)
            return ax.legend(handles=handles, **legend_kwargs)
        else:
            # No handles or labels, use default legend
            return ax.legend(**legend_kwargs)

    def _apply_unified_border_style(self, ax):
        """Apply unified border style to axes"""
        for spine in ax.spines.values():
            spine.set_linewidth(self.unified_border_style['linewidth'])
            spine.set_color(self.unified_border_style['color'])
        plt.rcParams['grid.linewidth'] = 0.5
        plt.rcParams['grid.alpha'] = 0.3

        # Use professional color theme
        sns.set_theme(style="whitegrid", palette="deep")

        self.source_colors = {
            'atmosphere': '#E74C3C', 'irrigation': '#3498DB',
            'fertilizer': '#9B59B6', 'manure': '#2ECC71'
        }
        self.metals = self.config.metals

        logging.info(f"Visualization suite initialized, charts will be saved to: {self.output_dir}")

    def _prepare_boxplot_data_by_source_type(self, df, value_column, source_type_column='source_type'):
        """
        Prepare data for box plot grouped by source type.

        This is a common pattern used in Figure 11 and Figure 12.

        Args:
            df: DataFrame containing the data
            value_column: Name of the column containing values to plot
            source_type_column: Name of the column containing source types

        Returns:
            Tuple of (box_data, box_labels, box_colors)
        """
        box_data = []
        box_labels = []
        box_colors = []

        for display_type in self.SOURCE_TYPE_ORDER:
            # Find all data source types that map to this display type
            matching_data = []

            # Check direct match
            if display_type in df[source_type_column].values:
                matching_data.append(df[df[source_type_column] == display_type])

            # Check mapped types
            for data_type, mapped_type in self.SOURCE_TYPE_MAPPING.items():
                if mapped_type == display_type and data_type in df[source_type_column].values:
                    matching_data.append(df[df[source_type_column] == data_type])

            # Combine all matching data
            if matching_data:
                source_data = pd.concat(matching_data, ignore_index=True)
                box_data.append(source_data[value_column].values)
                box_labels.append(f'{display_type.capitalize()}\n(n={len(source_data)})')
                box_colors.append(self.source_colors.get(display_type, '#666666'))
                logging.info(f"  {display_type}: {len(source_data)} data points")
            else:
                # If no data for this source type, add empty array
                box_data.append([])
                box_labels.append(f'{display_type.capitalize()}\n(n=0)')
                box_colors.append(self.source_colors.get(display_type, '#666666'))
                logging.info(f"  {display_type}: 0 data points (no data)")

        return box_data, box_labels, box_colors

    def generate_all_figures(self):
        """Generate all core visualization charts, including new source merging analysis charts."""
        logging.info("=" * 50)
        logging.info("GENERATING VISUALIZATION FIGURES")
        logging.info("=" * 50)
        figure_functions = {
            1: self.figure1_source_chemical_fingerprints,
            2: self.figure2_soil_correlation_heatmap,
            3: self.figure3_6_source_contribution_heatmaps,
            7: self.figure7_training_process,
            8: self.figure8_integrated_spatial_distribution,
            9: self.figure9_pmf_diagnostics_enhanced,
            10: self.figure10_contribution_barcharts,
            11: self.figure11_uncertainty_visualization,
            13: self.figure13_importance_map,
            14: self.figure14_integrated_spatial_map,
            15: self.figure15_parameter_sensitivity_analysis,
            16: self.figure16_dem_surface_3d
        }
        for num, func in figure_functions.items():
            try:
                logging.info(f"Generating Figure {num}...")
                func()
                logging.info(f"Figure {num} completed successfully")
            except Exception as e:
                logging.error(f"Error generating Figure {num}: {e}")
                import traceback
                traceback.print_exc()

        logging.info("=" * 50)
        logging.info("ALL VISUALIZATION FIGURES COMPLETED")
        logging.info("=" * 50)

    def _save_fig(self, fig, filename: str):
        """Save chart to file, compliant with international journal standards."""
        try:
            path = os.path.join(self.output_dir, filename)

            fig.savefig(path,
                       dpi=400,

                       bbox_inches='tight',
                       facecolor='white',
                       edgecolor='none',
                       format='png',
                       transparent=False,
                       pad_inches=0.1)

            plt.close(fig)
            logging.debug(f"Chart saved: {path}")
            return True
        except Exception as e:
            logging.error(f"Save failed {filename}: {e}")
            return False

    def _save_fig_high_res(self, fig, filename: str):
        """Save ultra-high resolution charts, dedicated to 3D visualization and other charts requiring high-quality output."""
        try:
            path = os.path.join(self.output_dir, filename)

            fig.savefig(path,
                       dpi=800,
                       bbox_inches='tight',
                       facecolor='white',
                       edgecolor='none',
                       format='png',
                       transparent=False,
                       pad_inches=0.1)

            plt.close(fig)
            logging.debug(f"Chart saved: {path}")
            return True
        except Exception as e:
            logging.error(f"Save failed {filename}: {e}")
            return False
            plt.close(fig)
            return False

    def _get_target_receptor_indices(self):
        """
        Determines the list of receptor indices to analyze based on user specification
        or a fallback mechanism.
        """
        all_receptor_indices = self.receptors_df.index.unique()

        user_specified_indices = [1, 10, 20, 40]

        target_indices = [idx for idx in user_specified_indices if idx in all_receptor_indices]

        if not target_indices:
            logging.info("Specified receptors (R1, R10, R20, R40) not found, selecting 4 receptors using arithmetic progression")
            if len(all_receptor_indices) >= 4:

                selected_indices = np.linspace(0, len(all_receptor_indices) - 1, 4, dtype=int)
                target_indices = all_receptor_indices[selected_indices].tolist()
            else:


                target_indices = all_receptor_indices.tolist()
            logging.info(f"Selected receptor indices: {target_indices}")
        else:
            logging.info(f"Found specified receptors, analyzing indices: {target_indices}")

        return target_indices

    def _get_target_receptor_indices_for_figure12(self):
        """
        Get 10 receptor indices for Figure 12 specifically.
        """
        all_receptor_indices = self.receptors_df.index.unique()

        if len(all_receptor_indices) >= 10:

            selected_indices = np.linspace(0, len(all_receptor_indices) - 1, 10, dtype=int)
            target_indices = all_receptor_indices[selected_indices].tolist()
        else:

            target_indices = all_receptor_indices.tolist()

        logging.info(f"Selected {len(target_indices)} receptor indices: {target_indices}")
        return target_indices

    def _get_top_nemerow_receptor_indices(self, top_n=10):
        """
        Get top N receptor indices by Nemerow index for Figure 8.
        """
        receptors_df_analyzed = self.analysis_results.get('receptors_with_nemerow')

        if receptors_df_analyzed is None or receptors_df_analyzed.empty:
            logging.warning("No Nemerow index data available, falling back to default selection")

            all_receptor_indices = self.receptors_df.index.unique()
            if len(all_receptor_indices) >= top_n:
                selected_indices = np.linspace(0, len(all_receptor_indices) - 1, top_n, dtype=int)
                target_indices = all_receptor_indices[selected_indices].tolist()
            else:
                target_indices = all_receptor_indices.tolist()
            return target_indices


        if 'nemerow_index' in receptors_df_analyzed.columns:
            top_receptors = receptors_df_analyzed.nlargest(top_n, 'nemerow_index')
            target_indices = top_receptors.index.tolist()

            logging.info(f"Selected {len(target_indices)} receptors with highest Nemerow index")
            logging.info(f"Receptor index: {target_indices}")
            return target_indices
        else:
            logging.warning("Nemerow index column not found, using default selection")
            return self._get_target_receptor_indices()
    def _smooth_lr_series(self, lr_values, ema_alpha: float = 0.15):
        """Return a smoothed learning-rate curve for visualization only.
        Strategy: piecewise-linear interpolation between actual change points;
        if fewer than 2 change points exist, fall back to EMA.
        This DOES NOT affect training; it only changes how the LR is drawn.
        """
        try:
            if not lr_values or len(lr_values) < 2:
                return lr_values
            # Identify indices where LR value actually changes
            anchors = [0]
            for i in range(1, len(lr_values)):
                if lr_values[i] != lr_values[i - 1]:
                    anchors.append(i)
            if anchors[-1] != len(lr_values) - 1:
                anchors.append(len(lr_values) - 1)

            # If LR never changes, use EMA smoothing (though it will remain flat)
            if len(anchors) <= 2 and lr_values.count(lr_values[0]) == len(lr_values):
                sm = []
                prev = float(lr_values[0])
                for v in lr_values:
                    prev = ema_alpha * float(v) + (1.0 - ema_alpha) * prev
                    sm.append(prev)
                return sm

            # Piecewise-linear interpolation between change points
            smoothed = [0.0] * len(lr_values)
            for a_idx in range(len(anchors) - 1):
                start = anchors[a_idx]
                end = anchors[a_idx + 1]
                v0 = float(lr_values[start])
                v1 = float(lr_values[end])
                span = max(1, end - start)
                for t in range(0, end - start + 1):
                    frac = t / span
                    smoothed[start + t] = v0 + (v1 - v0) * frac
            return smoothed
        except Exception:
            # Safety fallback: return raw series on any unexpected error
            return lr_values

    def _get_receptor_actual_names(self, receptor_indices):
        """
        Get actual receptor ID names (e.g., CJ14, GG3, etc.)
        """
        actual_names = []
        for idx in receptor_indices:
            if idx < len(self.receptors_df):

                if 'sample' in self.receptors_df.columns:
                    actual_name = self.receptors_df.iloc[idx]['sample']
                else:
                    actual_name = f'R{idx}'
                actual_names.append(actual_name)
            else:
                actual_names.append(f'R{idx}')
        return actual_names

    def figure1_source_chemical_fingerprints(self):
        source_types = self.sources_df['source_type'].unique()
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()
        fig.suptitle('Source Chemical Fingerprints', fontsize=20, fontweight='bold', family='Times New Roman')


        display_mapping = {
            'pesticide': 'fertilizer'
        }

        for i, source_type in enumerate(source_types[:4]):
            ax = axes[i]
            source_data = self.sources_df[self.sources_df['source_type'] == source_type]
            if not source_data.empty:
                metal_means = source_data[self.metals].mean()


                display_name = display_mapping.get(source_type, source_type)
                color = self.source_colors.get(display_name, self.source_colors.get(source_type, '#333333'))

                sns.barplot(x=metal_means.index, y=metal_means.values, ax=ax,
                            color=color, edgecolor='black', linewidth=0.7)
                ax.set_title(f'{display_name.capitalize()} (n={len(source_data)})', fontsize=14, weight='bold', family='Times New Roman', pad=10)
                ax.set_ylabel('Concentration (mg/kg/year)', fontsize=12, fontweight='bold', family='Times New Roman')
                ax.tick_params(axis='x', rotation=0, labelsize=11)
                ax.set_xlabel(None, fontweight='bold')

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        self._save_fig(fig, 'Figure1_Source_Chemical_Fingerprints.png')

    def figure2_soil_correlation_heatmap(self):
        correlation_matrix = self.receptors_df[self.metals].corr()
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='viridis',
                    square=True, linewidths=0.5, linecolor='white', cbar_kws={"shrink": .8}, ax=ax,
                    annot_kws={'fontsize': 10, 'family': 'Times New Roman'})
        ax.set_title('Heavy Metal Correlation Matrix in Receptors', fontsize=14, fontweight='bold', pad=10, family='Times New Roman')
        ax.set_xticklabels(ax.get_xticklabels(), fontsize=11, family='Times New Roman')
        ax.set_yticklabels(ax.get_yticklabels(), fontsize=11, family='Times New Roman')
        # Set colorbar font
        cbar = ax.collections[0].colorbar
        if cbar:
            cbar.ax.tick_params(labelsize=11)
            for label in cbar.ax.get_yticklabels():
                label.set_family('Times New Roman')
        self._save_fig(fig, 'Figure2_Soil_Correlation_Heatmap.png')

    def _create_contribution_matrix(self, source_type):
        col = 'contribution_percent_total'
        filtered_preds = self.predictions_df[self.predictions_df['source_type'] == source_type]
        if filtered_preds.empty: return None
        try:

            filtered_preds = filtered_preds.copy()
            filtered_preds['display_name'] = filtered_preds['source_sample']


            non_zero_preds = filtered_preds[filtered_preds[col] > 0]

            if non_zero_preds.empty:
                logging.warning(f"No non-zero contributions found for {source_type} source type")
                return None


            pivot = pd.pivot_table(non_zero_preds, values=col, index='receptor_idx', columns='display_name', fill_value=0)

            all_receptors = set(range(68))

            current_receptors = set(pivot.index)
            missing_receptors = all_receptors - current_receptors

            if missing_receptors:

                for receptor in missing_receptors:
                    pivot.loc[receptor] = 0.0


                pivot = pivot.sort_index()

                logging.info(f"{source_type}: Added {len(missing_receptors)} missing receptor zero-value rows")

            logging.info(f"{source_type}: Matrix dimensions {pivot.shape[0]} receptors × {pivot.shape[1]} sources (non-zero contributions only)")
            return pivot
        except Exception as e:
            logging.warning(f"Failed to create contribution matrix ({source_type}): {e}")
            import traceback
            logging.warning(f"Detailed error info: {traceback.format_exc()}")
            return None

    def figure3_6_source_contribution_heatmaps(self):
        """
        Enhanced source contribution heatmaps with the following new features:
        1. Display merged source names
        2. Add visual hints for uncertainty information
        3. Improved color mapping and annotations
        """
        source_types = self.predictions_df['source_type'].unique()
        for i, source_type in enumerate(source_types):
            contribution_matrix = self._create_contribution_matrix(source_type)
            if contribution_matrix is not None and not contribution_matrix.empty:

                receptor_idx_to_sample = {}
                if hasattr(self, 'receptors_df') and self.receptors_df is not None:
                    for idx, row in self.receptors_df.iterrows():
                        sample_name = row.get('sample', f'R{idx}')
                        receptor_idx_to_sample[idx] = sample_name
                else:

                    for idx in contribution_matrix.index:
                        receptor_idx_to_sample[idx] = f'R{idx}'


                contribution_matrix.index = [receptor_idx_to_sample.get(idx, f'R{idx}') for idx in contribution_matrix.index]

                figsize_width = max(20, len(contribution_matrix.columns) * 0.4)
                figsize_height = max(16, len(contribution_matrix.index) * 0.4)
                fig, ax = plt.subplots(figsize=(figsize_width, figsize_height))


                annot_matrix = contribution_matrix.copy()
                annot_matrix = annot_matrix.applymap(lambda x: f'{x:.1f}' if x > 0.0 else '')

                sns.heatmap(contribution_matrix, cmap="viridis", ax=ax,
                            vmin=0, vmax=100,
                            linewidths=0.8, linecolor='white', xticklabels=True, yticklabels=True,
                            cbar_kws={'label': 'Contribution Rate (%)', 'shrink': 0.8},
                            annot=annot_matrix, fmt='', annot_kws={'size': 8, 'family': 'Times New Roman'})

                # Set colorbar font
                cbar = ax.collections[0].colorbar
                if cbar:
                    cbar.set_label('Contribution Rate (%)', fontsize=20, fontweight='bold', family='Times New Roman')
                    cbar.ax.tick_params(labelsize=19)
                    for label in cbar.ax.get_yticklabels():
                        label.set_family('Times New Roman')

                n_sources = contribution_matrix.shape[1]
                n_receptors = contribution_matrix.shape[0]


                expected_sources = {
                    'atmosphere': 80,
                    'irrigation': 46,
                    'fertilizer': 60,
                    'manure': 27
                }

                expected_count = expected_sources.get(source_type, n_sources)
                source_coverage = n_sources / expected_count * 100
                receptor_coverage = n_receptors / 68 * 100

                logging.info(f"Figure {i+3} {source_type} heatmap data statistics:")
                logging.info(f"  Source count: {n_sources}/{expected_count} ({source_coverage:.1f}%)")
                logging.info(f"  Receptor count: {n_receptors}/68 ({receptor_coverage:.1f}%)")


                combined_count = 0
                if 'source_combined_name' in self.predictions_df.columns:
                    type_data = self.predictions_df[self.predictions_df['source_type'] == source_type]
                    combined_count = type_data['is_combined'].sum() if 'is_combined' in type_data.columns else 0

                title = f'{source_type.capitalize()} Source Contribution\n'
                title += f'Sources: {n_sources}/{expected_count} ({source_coverage:.1f}%), '
                title += f'Receptors: {n_receptors}/68 ({receptor_coverage:.1f}%)'

                if combined_count > 0:
                    title += f'\n(Including {combined_count} Combined Sources)'

                ax.set_title(title, fontsize=40, fontweight='bold', pad=10, family='Times New Roman')
                ax.set_xlabel('Source Samples', fontsize=40, fontweight='bold', family='Times New Roman')
                ax.set_ylabel('Receptor Samples', fontsize=40, fontweight='bold', family='Times New Roman')


                plt.setp(ax.get_yticklabels(), rotation=0, fontsize=19, family='Times New Roman')
                plt.setp(ax.get_xticklabels(), rotation=45, ha='right', fontsize=19, family='Times New Roman')


                ax.grid(True, alpha=0.3)

                fig.tight_layout()
                self._save_fig(fig, f'Figure{i+3}_{source_type}_Contribution_Heatmap.png')

    def figure7_training_process(self):
        """Plot training process curves for 4 loss functions: Chemistry/Strength/Distance/Cosine."""
        history = self.model_data.get('training_history')
        if not history:
            print("  - Training history data not found, skipping Figure 7 generation.")
            return


        if not any(key in history and history[key] for key in ['chemical_loss', 'strength_loss', 'distance_loss']):
            print("  - Key loss data not found, skipping Figure 7 generation.")
            return

        epochs = range(1, len(history['chemical_loss']) + 1)

        if 'total_loss' not in history or not history['total_loss']:
            total_loss = []
            for i in range(len(history['chemical_loss'])):
                total = (history['chemical_loss'][i] +
                        history['strength_loss'][i] +
                        history['distance_loss'][i] +
                        (history.get('reconstruction_loss', [0]*len(history['chemical_loss']))[i]))
                total_loss.append(total)
            history['total_loss'] = total_loss


        items = [
            ('chemical_loss',   'Figure7a_Chemical_Loss.png',   'Chemical Loss Vs Epochs',  '#F1C40F'),
            ('strength_loss',   'Figure7b_Strength_Loss.png',   'Strength Loss Vs Epochs',  '#2ECC71'),
            ('distance_loss',   'Figure7c_Distance_Loss.png',   'Distance Loss Vs Epochs',  '#9B59B6'),
            ('reconstruction_loss', 'Figure7d_Reconstruction_Loss.png', 'Reconstruction Loss Vs Epochs', '#1ABC9C'),
            ('total_loss',      'Figure7e_Total_Loss.png',      'Total Loss Vs Epochs',     '#E74C3C'),
        ]
        for key, fname, title, color in items:
            if key not in history or not history[key]:
                continue
            fig, ax1 = plt.subplots(figsize=(10, 6))
            ax1.plot(epochs, history[key], color=color, linewidth=2, label=key.replace('_',' ').title())
            ax1.set_xlabel('Epoch', fontsize=12, family='Times New Roman')
            ax1.set_ylabel('Loss Value', fontsize=12, family='Times New Roman')
            ax1.grid(True, which="both", ls="--", linewidth=0.5)
            ax1.tick_params(axis='both', labelsize=11)


            for spine in ax1.spines.values():
                spine.set_color('black')
                spine.set_linewidth(1.0)

            ax2 = ax1.twinx()
            if 'learning_rate' in history and history['learning_rate']:
                # Use original learning rate data to show true step-wise changes (consistent with Figure 7f)
                lr_data = history['learning_rate']
                # Ensure learning rate data matches epochs length
                if len(lr_data) >= len(epochs):
                    lr_epochs = epochs
                    lr_values = lr_data[:len(epochs)]
                else:
                    lr_epochs = range(1, len(lr_data) + 1)
                    lr_values = lr_data
                ax2.plot(lr_epochs, lr_values, color='#7f7f7f', linestyle='-', label='Learning Rate')
            ax2.set_ylabel('Learning Rate', fontsize=12, family='Times New Roman')
            ax2.tick_params(axis='y', labelsize=11)


            for spine in ax2.spines.values():
                spine.set_color('black')
                spine.set_linewidth(1.0)


            lines, labels = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            if lines2:
                ax2.legend(lines + lines2, labels + labels2, loc='upper right', prop={'family': 'Times New Roman', 'size': 11})
            else:
                ax1.legend(loc='upper right', prop={'family': 'Times New Roman', 'size': 11})

            fig.suptitle(title, fontsize=16, fontweight='bold', family='Times New Roman')
            fig.tight_layout()
            self._save_fig(fig, fname)

        # Note: Composite Figure 7_Training_Losses.png has been removed as redundant.
        # Individual panels (7a–7e) and the enhanced style figure (7x) remain.
        self.figure7x_loss_weight_evolution_style()

        # Generate Figure 7f: Adaptive weights evolution
        self.figure7f_adaptive_weights_evolution()



    def figure8_integrated_spatial_distribution(self):
        """Integrated Spatial Distribution of High-Pollution Receptors and Connected Sources"""
        top_contributions = self.analysis_results.get('top5_contribution_report')
        if top_contributions is None or top_contributions.empty:
            logging.warning("Top 5 contribution report not found or is empty, skipping Figure 8")
            return

        if 'receptor_idx' not in top_contributions.columns:
            logging.warning("'receptor_idx' column not found, skipping Figure 8")
            return

        receptors_df_analyzed = self.analysis_results.get('receptors_with_nemerow')
        if receptors_df_analyzed is None:
            logging.warning("Receptors with Nemerow index not found, skipping Figure 8")
            return

        target_indices = self._get_top_nemerow_receptor_indices(top_n=10)
        logging.info(f"Selected 10 receptors with highest Nemerow index: {target_indices}")

        fig, ax = plt.subplots(figsize=(16, 12))

        # Determine which column carries contribution values
        contrib_col = None
        for col in ['contribution_percent_total', 'contribution_percent_mean', 'contribution_percent', 'contribution']:
            if col in top_contributions.columns:
                contrib_col = col
                break

        # Line width scaling parameters
        min_lw, max_lw, gamma = 0.8, 6.0, 1.8

        all_connected_sources = set()
        receptor_source_contribs = {}

        for target_receptor_idx in target_indices:
            receptor_contributions = top_contributions[top_contributions['receptor_idx'] == target_receptor_idx]
            if not receptor_contributions.empty:
                contrib_map = {}
                for _, row in receptor_contributions.iterrows():
                    s_idx = int(row['source_idx'])
                    all_connected_sources.add(s_idx)
                    if contrib_col is not None:
                        val = float(row[contrib_col])
                        if contrib_col == 'contribution' and val <= 1.0:
                            val *= 100.0
                        val = max(0.0, min(100.0, val))
                    else:
                        val = 0.0
                    contrib_map[s_idx] = val
                receptor_source_contribs[target_receptor_idx] = contrib_map

        # Draw connected sources
        for source_idx in all_connected_sources:
            if source_idx < len(self.sources_df):
                source_info = self.sources_df.iloc[source_idx]
                source_type = source_info['source_type']
                color = self.source_colors.get(source_type, 'gray')
                ax.scatter(source_info['lon'], source_info['lat'],
                          c=color, s=80, alpha=0.7,
                          edgecolors='black', linewidth=0.5, zorder=2)

        # Draw connection lines
        for target_receptor_idx in target_indices:
            if target_receptor_idx < len(receptors_df_analyzed):
                receptor_info = receptors_df_analyzed.iloc[target_receptor_idx]
                if target_receptor_idx in receptor_source_contribs:
                    for source_idx, perc in receptor_source_contribs[target_receptor_idx].items():
                        if source_idx < len(self.sources_df):
                            source_info = self.sources_df.iloc[source_idx]
                            v01 = (max(0.0, min(100.0, perc)) / 100.0) ** gamma
                            lw = min_lw + (max_lw - min_lw) * v01
                            alpha = 0.25 + 0.65 * v01
                            ax.plot([receptor_info['lon'], source_info['lon']],
                                    [receptor_info['lat'], source_info['lat']],
                                    color='black', alpha=alpha, linewidth=lw, zorder=1)

        # Draw high-pollution receptors
        for target_receptor_idx in target_indices:
            if target_receptor_idx < len(receptors_df_analyzed):
                receptor_info = receptors_df_analyzed.iloc[target_receptor_idx]
                ax.scatter(receptor_info['lon'], receptor_info['lat'],
                          c='red', s=400, marker='*',
                          edgecolors='black', linewidth=2,
                          alpha=0.9, zorder=5)
                ax.annotate(f'R{target_receptor_idx}',
                           (receptor_info['lon'], receptor_info['lat']),
                           xytext=(8, 8), textcoords='offset points',
                           fontsize=11, fontweight='bold',
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.5),  # Semi-transparent label box
                           zorder=6)

        ax.set_xlabel('Longitude', fontsize=22, fontweight='bold', family='Times New Roman')
        ax.set_ylabel('Latitude', fontsize=22, fontweight='bold', family='Times New Roman')
        ax.set_title('Integrated Spatial Distribution Of High-Pollution Receptors And Connected Sources',
                    fontsize=24, fontweight='bold', family='Times New Roman', pad=10)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(102.5, 102.95)
        ax.set_ylim(29.15, 29.7)
        ax.tick_params(axis='both', labelsize=13)
        ax.set_xticklabels(ax.get_xticklabels(), fontsize=13, family='Times New Roman')
        ax.set_yticklabels(ax.get_yticklabels(), fontsize=13, family='Times New Roman')

        # Legend
        legend_elements = [Line2D([0], [0], marker='*', color='w', label='High-Pollution Receptors',
                                 markerfacecolor='red', markersize=15, markeredgecolor='black')]
        for source_type, color in self.source_colors.items():
            legend_elements.append(Line2D([0], [0], marker='o', color='w', label=f'{source_type.capitalize()} Sources',
                                          markerfacecolor=color, markersize=10, markeredgecolor='black'))

        # Apply unified legend style
        # Ensure legend frame uses pure black edge
        leg = self._apply_unified_legend_style(ax, handles=legend_elements, title='Legend', title_fontsize=12, fontsize=13)
        if leg is not None:
            frame = leg.get_frame()
            frame.set_edgecolor('black')  # Use pure black edge per user request
            # Set legend font
            for text in leg.get_texts():
                text.set_family('Times New Roman')
                text.set_fontsize(13)
            leg.get_title().set_family('Times New Roman')
            leg.get_title().set_fontsize(12)

        plt.tight_layout()
        self._save_fig(fig, 'Figure8_Integrated_Spatial_Distribution.png')

    def figure9_pmf_diagnostics_enhanced(self):
        """Enhanced PMF Diagnostics Plot"""
        diag = self.model_data.get('pmf_diagnostics')
        if not diag:
            return

        n_factors, errors, explained_variance = diag['n_factors'], diag['reconstruction_error'], diag['explained_variance']
        fig, ax1 = plt.subplots(figsize=(12, 7))

        # Enhanced color scheme
        color1 = '#2E86AB'  # Professional blue
        color2 = '#F18F01'  # Professional orange

        ax1.set_xlabel('Number of Factors', fontsize=16, fontweight='bold', family='Times New Roman')
        ax1.set_ylabel('Reconstruction Error', fontsize=16, color=color1, fontweight='bold', family='Times New Roman')
        line1 = ax1.plot(n_factors, errors, marker='o', linestyle='-', color=color1,
                        linewidth=2.5, markersize=8, markeredgecolor='black', markeredgewidth=1.5)
        ax1.tick_params(axis='y', labelcolor=color1, labelsize=13)
        ax1.tick_params(axis='x', labelsize=13)
        ax1.set_xticklabels(ax1.get_xticklabels(), fontsize=13, family='Times New Roman')
        ax1.set_yticklabels(ax1.get_yticklabels(), fontsize=13, family='Times New Roman')
        ax1.grid(True, alpha=0.3)

        ax2 = ax1.twinx()
        ax2.set_ylabel('Explained Variance (%)', fontsize=16, color=color2, fontweight='bold', family='Times New Roman')
        line2 = ax2.plot(n_factors, explained_variance, marker='s', linestyle='--', color=color2,
                        linewidth=2.5, markersize=8, markeredgecolor='black', markeredgewidth=1.5)
        ax2.tick_params(axis='y', labelcolor=color2, labelsize=13)
        ax2.set_yticklabels(ax2.get_yticklabels(), fontsize=13, family='Times New Roman')

        # Enhanced title
        ax1.set_title('Enhanced PMF Diagnostics: Model Selection Criteria',
                     fontsize=18, fontweight='bold', pad=10, family='Times New Roman')

        # Enhanced legend - Fix: Apply unified legend style and position
        lines = line1 + line2
        labels = ['Reconstruction Error', 'Explained Variance (%)']
        self._apply_unified_legend_style(ax1, handles=lines, labels=labels, fontsize=12)

        # Enhanced styling - Fix: Apply unified border style
        self._apply_unified_border_style(ax1)
        self._apply_unified_border_style(ax2)

        # Set right y-axis range to [0.4, 1.2] for Figure 9
        try:
            max_ev = float(np.nanmax(explained_variance)) if len(explained_variance) > 0 else None
            if max_ev is not None:
                if max_ev <= 1.000001:
                    ax2.set_ylim(0.4, 1.2)  # Changed from [0.0, 1.05] to [0.4, 1.2]
                else:
                    ax2.set_ylim(0.0, 105.0)
        except Exception:
            pass  # Keep default limits on any error

        plt.tight_layout()
        self._save_fig(fig, 'Figure9_Enhanced_PMF_Diagnostics.png')




    def figure10_contribution_barcharts(self):

        pivot_df = self.analysis_results.get('top5_contribution_pivot_summary')
        if pivot_df is None:
            logging.warning("all source contribution pivot summary not found, falling back to original data")
            pivot_df = self.analysis_results.get('contribution_pivot_summary')
            if pivot_df is None:
                logging.warning("contribution_pivot_summary not found in analysis_results")
                return

        receptors_analyzed = self.analysis_results.get('receptors_with_nemerow')
        if receptors_analyzed is None:
            logging.warning("receptors_with_nemerow not found in analysis_results")
            return


        target_indices = self._get_top_nemerow_receptor_indices(top_n=10)

        if isinstance(pivot_df.index, pd.MultiIndex):

            plot_df = pivot_df[pivot_df.index.get_level_values('receptor_idx').isin(target_indices)].copy()
        else:

            if 'receptor_sample' not in pivot_df.columns:
                pivot_df['receptor_sample'] = pivot_df['receptor_idx'].apply(lambda x: f'R{x}')

            # Ensure receptor_sample column exists in receptors_analyzed
            if 'receptor_sample' not in receptors_analyzed.columns:
                receptors_analyzed['receptor_sample'] = receptors_analyzed.index.map(lambda x: f'R{x}')

            target_receptor_samples = receptors_analyzed[receptors_analyzed.index.isin(target_indices)]['receptor_sample']
            plot_df = pivot_df[pivot_df['receptor_sample'].isin(target_receptor_samples)].copy()

        if plot_df.empty:
            logging.warning("No data available for Figure 12 after filtering for target receptors")
            return

        if not isinstance(plot_df.index, pd.MultiIndex):

            plot_df['receptor_sample'] = pd.Categorical(plot_df['receptor_sample'], categories=target_receptor_samples, ordered=True)
            plot_df = plot_df.sort_values('receptor_sample')
            plot_df.set_index(['receptor_idx', 'receptor_sample'], inplace=True)

        source_cols = [col for col in plot_df.columns if col in self.source_colors]

        if not source_cols:
            logging.warning("No source contribution columns found to plot for Figure 12")
            return


        self._create_comprehensive_contribution_barchart(plot_df, source_cols)

    def _create_comprehensive_contribution_barchart(self, plot_df, source_cols):

        expected_source_types = ['atmosphere', 'irrigation', 'fertilizer', 'manure']

        for source_type in expected_source_types:
            if source_type not in plot_df.columns:
                plot_df[source_type] = 0.0
                logging.info(f"Added missing source type '{source_type}' with 0% contribution")

        source_cols_ordered = [col for col in expected_source_types if col in plot_df.columns]

        fig, ax = plt.subplots(figsize=(14, 8))

        display_mapping = {
            'pesticide': 'fertilizer'
        }

        display_colors = []
        display_labels = []
        for col in source_cols_ordered:
            display_name = display_mapping.get(col, col)
            display_labels.append(display_name.capitalize())
            display_colors.append(self.source_colors.get(display_name, self.source_colors.get(col, '#666666')))

        plot_df[source_cols_ordered].plot(kind='bar', stacked=True, ax=ax,
                                         color=display_colors,
                                         width=0.8, edgecolor='black', linewidth=0.5)

        ax.set_title('All Source Contribution Analysis For Top 10 Receptors By Nemerow Index',
                     fontsize=20, fontweight='bold', pad=10, family='Times New Roman')
        ax.set_xlabel('Receptor (Ranked by Nemerow Index)', fontsize=16, fontweight='bold', family='Times New Roman')
        ax.set_ylabel('Contribution (%)', fontsize=16, fontweight='bold', family='Times New Roman')

        ax.set_xticklabels([f'{i[1]}' for i in plot_df.index],
                           rotation=45, ha='right', fontsize=13, family='Times New Roman')
        ax.tick_params(axis='y', labelsize=13)
        ax.set_yticklabels(ax.get_yticklabels(), fontsize=13, family='Times New Roman')

        ax.legend(display_labels, title='Source Type', loc='upper right',
                 fontsize=11, title_fontsize=12, frameon=True, fancybox=True, shadow=True,
                 prop={'family': 'Times New Roman'})

        # Remove grid lines as requested by user
        ax.set_ylim(0, 100)

        # Enhance coordinate axis lines visual effects for Figure 10
        for spine in ax.spines.values():
            spine.set_edgecolor('black')
            spine.set_linewidth(1.0)
            spine.set_alpha(1.0)

        for i, (_, row) in enumerate(plot_df.iterrows()):
            cumulative = 0
            total_for_receptor = row[source_cols_ordered].sum()

            receptor_name = plot_df.index[i][1]
            if abs(total_for_receptor - 100.0) > 0.1:
                logging.warning(f"Figure 10 - Receptor {receptor_name}: contributions sum to {total_for_receptor:.2f}% (expected 100%)")

            for col in source_cols_ordered:
                value = row[col]
                if value > 3:  # Show percentage for values > 3% to avoid clutter
                    ax.text(i, cumulative + value/2, f'{value:.1f}%',
                           ha='center', va='center', fontsize=9, fontweight='bold', color='white', family='Times New Roman')
                cumulative += value

        plt.tight_layout()
        self._save_fig(fig, 'Figure10_All_Source_Contribution_Analysis.png')





    def generate_enhanced_charts(self):
        """
        Generate enhanced static charts
        """
        logging.info("=== Generating enhanced static charts ===")
        self._generate_source_comparison_chart()

        logging.info("Enhanced static chart generation completed")

    def _generate_source_comparison_chart(self):
        """Generate source contribution comparison chart"""
        try:

            source_summary = self.predictions_df.groupby('source_type').agg({
                'contribution_percent_total': ['mean', 'std', 'count']
            }).round(2)

            source_summary.columns = ['Mean_Contribution', 'Std_Deviation', 'Sample_Count']
            source_summary = source_summary.reset_index()

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))


            bars1 = ax1.bar(source_summary['source_type'], source_summary['Mean_Contribution'],
                           yerr=source_summary['Std_Deviation'], capsize=5,
                           color=[self.source_colors.get(st, 'gray') for st in source_summary['source_type']],
                           alpha=0.8, edgecolor='black')

            ax1.set_title('Average Contribution Rate by Source Type', fontsize=16, fontweight='bold', pad=10)
            ax1.set_ylabel('Average Contribution Rate (%)', fontsize=12)
            ax1.set_xlabel('Source Type', fontsize=12)


            for i, bar in enumerate(bars1):
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height + source_summary.iloc[i]['Std_Deviation'],
                        f'{height:.1f}%', ha='center', va='bottom', fontweight='bold')


            ax2.pie(source_summary['Sample_Count'], labels=source_summary['source_type'],
                   colors=[self.source_colors.get(st, 'gray') for st in source_summary['source_type']],
                   autopct='%1.1f%%', startangle=90)
            ax2.set_title('Sample Distribution by Source Type', fontsize=14, fontweight='bold', pad=10)

            plt.tight_layout()
            self._save_fig(fig, 'Enhanced_Source_Comparison.png')

        except Exception as e:
            logging.error(f"Error generating source comparison chart: {e}")




    def figure11_uncertainty_visualization(self):
        """Create box-scatter plot visualization showing uncertainty distribution by source type.

        Uses real standard deviation data from ensemble predictions to quantify uncertainty.
        X-axis: 4 source types (fertilizer, manure, industrial, atmospheric)
        Y-axis: Uncertainty percentage (dynamic range based on actual data)
        Visualization: Box plot with overlaid scatter points
        """
        try:
            # Get predictions data with uncertainty metrics
            if self.predictions_df is None or self.predictions_df.empty:
                logging.warning("No predictions data available for uncertainty visualization")
                # Try to get data from alternative sources
                if self.analysis_results and 'contribution_detailed_summary' in self.analysis_results:
                    self.predictions_df = self.analysis_results['contribution_detailed_summary']
                    logging.info(f"Found alternative data source with shape: {self.predictions_df.shape}")
                else:
                    return

            # Check for standard deviation column (real uncertainty data from ensemble)
            # This is the contribution_percent_std from Final_Ensemble_Report.xlsx
            std_col = None
            for col in ['contribution_percent_std', 'contribution_std']:
                if col in self.predictions_df.columns:
                    std_col = col
                    break

            if std_col is None:
                logging.warning("No standard deviation column found in predictions data - cannot compute uncertainty")
                return


            # Extract all source-receptor relationships with their uncertainty data
            # Use real standard deviation from ensemble predictions (contribution_percent_std)
            logging.info(f"Processing {len(self.predictions_df)} source-receptor relationships for uncertainty visualization")
            logging.info(f"Using uncertainty column: {std_col}")

            # Prepare data for box-scatter plot
            # Directly use contribution_percent_std values (already in percentage)
            uncertainty_data = []
            for idx, row in self.predictions_df.iterrows():
                # Get source type information
                source_type = row.get('source_type', 'unknown')

                # Get real standard deviation (uncertainty metric)
                # This is contribution_percent_std - already in percentage form
                std_value = row[std_col]

                # Skip invalid data
                if pd.isna(std_value) or std_value < 0:
                    continue

                uncertainty_data.append({
                    'index': idx,
                    'receptor_idx': row.get('receptor_idx', -1),
                    'source_idx': row.get('source_idx', -1),
                    'source_type': source_type,
                    'uncertainty_percent': std_value  # Direct use of std value
                })

            if not uncertainty_data:
                logging.warning("No uncertainty data available to plot")
                return

            uncertainty_df = pd.DataFrame(uncertainty_data)

            # Calculate overall statistics for dynamic y-axis range
            min_uncertainty = uncertainty_df['uncertainty_percent'].min()
            max_uncertainty = uncertainty_df['uncertainty_percent'].max()
            mean_uncertainty = uncertainty_df['uncertainty_percent'].mean()
            median_uncertainty = uncertainty_df['uncertainty_percent'].median()

            # Log overall statistics
            logging.info(f"\n=== Uncertainty Data Statistics ===")
            logging.info(f"Total source-receptor relationships: {len(uncertainty_df)}")
            logging.info(f"Overall Min: {min_uncertainty:.4f}%")
            logging.info(f"Overall Max: {max_uncertainty:.4f}%")
            logging.info(f"Overall Mean: {mean_uncertainty:.4f}%")
            logging.info(f"Overall Median: {median_uncertainty:.4f}%")

            # Log statistics by source type
            logging.info(f"\n=== Statistics by Source Type ===")
            for source_type in sorted(uncertainty_df['source_type'].unique()):
                source_data = uncertainty_df[uncertainty_df['source_type'] == source_type]
                logging.info(f"  {source_type}: n={len(source_data)}, "
                           f"mean={source_data['uncertainty_percent'].mean():.4f}%, "
                           f"median={source_data['uncertainty_percent'].median():.4f}%, "
                           f"max={source_data['uncertainty_percent'].max():.4f}%")

            # Fixed y-axis range: 0~5%
            y_max = 5.0

            # Check if data exceeds the fixed range
            if max_uncertainty > y_max:
                logging.warning(f"\n=== WARNING: Data exceeds Y-axis range ===")
                logging.warning(f"Data max value: {max_uncertainty:.4f}% exceeds y-axis limit of {y_max}%")
                logging.warning(f"Some data points will be clipped in the visualization")

            logging.info(f"\n=== Y-axis Range Selection ===")
            logging.info(f"Data max value: {max_uncertainty:.4f}%")
            logging.info(f"Fixed y-axis range: 0 to {y_max}%")
            logging.info(f"Y-axis ticks: 0%, 1%, 2%, 3%, 4%, 5%")

            # Create box-scatter plot
            fig, ax = plt.subplots(figsize=(14, 8))

            # Prepare data for box plot using common method
            logging.info(f"\n=== Source Type Mapping ===")
            logging.info(f"Display order: {self.SOURCE_TYPE_ORDER}")
            logging.info(f"Data source types found: {sorted(uncertainty_df['source_type'].unique())}")

            box_data, box_labels, box_colors = self._prepare_boxplot_data_by_source_type(
                df=uncertainty_df,
                value_column='uncertainty_percent',
                source_type_column='source_type'
            )

            # Create box plots
            bp = ax.boxplot(box_data,
                           labels=box_labels,
                           patch_artist=True,
                           widths=0.6,
                           showmeans=True,
                           meanprops=dict(marker='D', markerfacecolor='red', markeredgecolor='black', markersize=8),
                           medianprops=dict(color='black', linewidth=2),
                           boxprops=dict(linewidth=1.5),
                           whiskerprops=dict(linewidth=1.5),
                           capprops=dict(linewidth=1.5))

            # Color the boxes
            for patch, color in zip(bp['boxes'], box_colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.6)

            # Overlay scatter points with jitter
            np.random.seed(42)  # For reproducible jitter
            for i, (data, color) in enumerate(zip(box_data, box_colors)):
                if len(data) > 0:
                    # Add jitter to x-coordinates
                    x_jitter = np.random.normal(i + 1, 0.04, size=len(data))
                    ax.scatter(x_jitter, data,
                             c=color,
                             s=15,
                             alpha=0.4,
                             edgecolors='black',
                             linewidths=0.3,
                             zorder=3)

            # Add horizontal reference lines at 1%, 2%, 3%
            ax.axhline(y=1, color='green', linestyle='--', alpha=0.5, linewidth=1.5, label='1%')
            ax.axhline(y=2, color='orange', linestyle='--', alpha=0.5, linewidth=1.5, label='2%')
            ax.axhline(y=3, color='red', linestyle='--', alpha=0.5, linewidth=1.5, label='3%')

            # Set labels and title
            ax.set_xlabel('Source Type', fontsize=16, fontweight='bold', family='Times New Roman')
            ax.set_ylabel('Uncertainty (Contribution Std Dev %)', fontsize=16, fontweight='bold', family='Times New Roman')
            ax.set_title(f'Uncertainty Distribution By Source Type (Total n={len(uncertainty_df)})',
                        fontsize=20, fontweight='bold', family='Times New Roman', pad=10)

            # Set fixed y-axis range: 0~5%
            ax.set_ylim(0, y_max)
            ax.set_yticks([0, 1, 2, 3, 4, 5])
            ax.set_yticklabels(['0%', '1%', '2%', '3%', '4%', '5%'], fontsize=13, family='Times New Roman')
            ax.tick_params(axis='x', labelsize=13)
            ax.set_xticklabels(box_labels, fontsize=13, family='Times New Roman')

            # Add grid
            ax.grid(True, alpha=0.3, linestyle='--', axis='y')

            # Add legend for reference lines
            from matplotlib.lines import Line2D
            legend_elements = [
                Line2D([0], [0], color='green', linestyle='--', linewidth=1.5, label='1% threshold'),
                Line2D([0], [0], color='orange', linestyle='--', linewidth=1.5, label='2% threshold'),
                Line2D([0], [0], color='red', linestyle='--', linewidth=1.5, label='3% threshold'),
            ]
            leg = ax.legend(handles=legend_elements, loc='upper right', fontsize=11, frameon=True, fancybox=True, shadow=True,
                           prop={'family': 'Times New Roman'})

            # Add overall statistics text box
            stats_text = (f'Overall Statistics:\n'
                         f'Min: {min_uncertainty:.4f}%\n'
                         f'Max: {max_uncertainty:.4f}%\n'
                         f'Mean: {mean_uncertainty:.4f}%\n'
                         f'Median: {median_uncertainty:.4f}%\n'
                         f'Std Dev: {uncertainty_df["uncertainty_percent"].std():.4f}%\n'
                         f'Y-axis range: 0-{y_max}%')

            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                   fontsize=10, verticalalignment='top', family='Times New Roman',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))

            plt.tight_layout()
            self._save_fig(fig, 'Figure11_Uncertainty_BoxScatter_by_SourceType.png')

            logging.info(f"Figure 11 completed successfully with y-axis range 0-{y_max}%")

        except Exception as e:
            logging.error(f"Error in Figure 11: {e}")
            import traceback
            traceback.print_exc()


    def figure13_importance_map(self):
        """Create importance map showing the importance of source points for receptor predictions."""
        if self.predictions_df is None or self.predictions_df.empty:
            logging.warning("No predictions data available for importance map")
            return


        importance_data = []

        for source_idx in self.predictions_df['source_idx'].unique():
            source_data = self.predictions_df[self.predictions_df['source_idx'] == source_idx]


            importance_score = source_data['contribution_percent_total'].mean()

            if source_idx < len(self.sources_df):
                source_info = self.sources_df.iloc[source_idx]
                importance_data.append({
                    'source_idx': source_idx,
                    'lon': source_info['lon'],
                    'lat': source_info['lat'],
                    'source_type': source_info['source_type'],
                    'importance_score': importance_score
                })

        if not importance_data:
            logging.warning("No importance data available to plot")
            return

        importance_df = pd.DataFrame(importance_data)


        fig, ax = plt.subplots(figsize=(12, 10))


        ax.scatter(self.receptors_df['lon'], self.receptors_df['lat'],
                  c='lightgray', s=30, alpha=0.6, label='Receptors', marker='s')


        source_markers = {
            'atmosphere': 'o',
            'irrigation': 's',
            'fertilizer': '^',
            'manure': 'D'
        }


        for source_type in importance_df['source_type'].unique():
            type_data = importance_df[importance_df['source_type'] == source_type]

            uniform_size = 60
            marker = source_markers.get(source_type, 'o')

            scatter = ax.scatter(type_data['lon'], type_data['lat'],
                               c=type_data['importance_score'],
                               s=uniform_size,
                               alpha=0.8,
                               cmap='viridis',
                               edgecolors='black',
                               linewidth=0.5,
                               marker=marker,
                               label=f'{source_type} Sources')


        cbar = plt.colorbar(scatter, ax=ax, shrink=0.8, aspect=20)
        cbar.set_label('Importance Score (%)', fontsize=18, fontweight='bold', family='Times New Roman')
        cbar.ax.tick_params(labelsize=13)
        for label in cbar.ax.get_yticklabels():
            label.set_family('Times New Roman')


        ax.set_xlim(102.5, 102.95)
        ax.set_ylim(29.15, 29.7)

        ax.set_xlabel('Longitude', fontsize=18, fontweight='bold', family='Times New Roman')
        ax.set_ylabel('Latitude', fontsize=18, fontweight='bold', family='Times New Roman')
        ax.set_title('Spatial Distribution Of Global Importance Scores For Individual Pollution Sources',
                    fontsize=20, fontweight='bold', family='Times New Roman', pad=10)
        ax.tick_params(labelsize=13)
        ax.set_xticklabels(ax.get_xticklabels(), fontsize=13, family='Times New Roman')
        ax.set_yticklabels(ax.get_yticklabels(), fontsize=13, family='Times New Roman')


        from matplotlib.lines import Line2D
        import matplotlib.cm as cm

        viridis_cmap = cm.get_cmap('viridis')
        min_color = viridis_cmap(0.0)

        legend_elements = [Line2D([0], [0], marker='s', color='w', markerfacecolor='lightgray',
                                 markersize=8, label='Receptors', markeredgecolor='black')]


        source_markers = {
            'atmosphere': 'o',
            'irrigation': 's',
            'fertilizer': '^',
            'manure': 'D'
        }

        for source_type in importance_df['source_type'].unique():
            marker = source_markers.get(source_type, 'o')
            legend_elements.append(Line2D([0], [0], marker=marker, color='w',
                                        markerfacecolor='lightgray', markersize=8,
                                        label=f'{source_type.capitalize()} Sources',
                                        markeredgecolor='black'))

        leg = ax.legend(handles=legend_elements, loc='upper right', frameon=True, fancybox=True, shadow=True,
                  facecolor='white', framealpha=0.95, edgecolor='#BDC3C7', prop={'family': 'Times New Roman', 'size': 11})
        ax.grid(True, alpha=0.3)

        # Set coordinate axis lines to pure black for Figure 13
        for spine in ax.spines.values():
            spine.set_edgecolor('black')
            spine.set_linewidth(1.5)
            spine.set_alpha(1.0)

        plt.tight_layout()
        self._save_fig(fig, 'Figure13_Importance_Map.png')

    def figure14_integrated_spatial_map(self):
        """Create integrated spatial map combining Figure 8 and 9 content: receptors colored by Nemerow index, sources distinguished by gray color and shape type."""

        receptors_df_analyzed = self.analysis_results.get('receptors_with_nemerow')
        if receptors_df_analyzed is None:
            logging.warning("No Nemerow index data available for integrated spatial map")
            return


        fig, ax = plt.subplots(figsize=(12, 10))


        receptor_scatter = ax.scatter(receptors_df_analyzed['lon'], receptors_df_analyzed['lat'],
                                    c=receptors_df_analyzed['nemerow_index'],
                                    s=60,
                                    cmap='viridis',
                                    alpha=0.8,
                                    edgecolors='black',
                                    linewidth=0.5,
                                    marker='s',
                                    label='Receptors')


        cbar_receptor = plt.colorbar(receptor_scatter, ax=ax, shrink=0.8, aspect=20)
        cbar_receptor.set_label('Nemerow Pollution Index', fontsize=12, fontweight='bold', family='Times New Roman')
        cbar_receptor.ax.tick_params(labelsize=11)
        for label in cbar_receptor.ax.get_yticklabels():
            label.set_family('Times New Roman')


        source_markers = {
            'atmosphere': 'o',
            'irrigation': 's',
            'fertilizer': '^',
            'manure': 'D'
        }


        for source_type in self.sources_df['source_type'].unique():
            type_data = self.sources_df[self.sources_df['source_type'] == source_type]
            marker = source_markers.get(source_type, 'o')

            ax.scatter(type_data['lon'], type_data['lat'],
                      c='lightgray',
                      s=30,
                      alpha=0.6,
                      edgecolors='none',
                      marker=marker,
                      label=f'{source_type.capitalize()} Sources')


        ax.set_xlim(102.5, 102.95)
        ax.set_ylim(29.15, 29.7)


        ax.set_xlabel('Longitude', fontsize=18, fontweight='bold', family='Times New Roman')
        ax.set_ylabel('Latitude', fontsize=18, fontweight='bold', family='Times New Roman')
        ax.set_title('Integrated Spatial Map: Receptors (Pollution Index) And Sources (By Type)',
                    fontsize=18, fontweight='bold', family='Times New Roman', pad=10)
        ax.tick_params(labelsize=13)
        ax.set_xticklabels(ax.get_xticklabels(), fontsize=13, family='Times New Roman')
        ax.set_yticklabels(ax.get_yticklabels(), fontsize=13, family='Times New Roman')


        from matplotlib.lines import Line2D
        legend_elements = [Line2D([0], [0], marker='s', color='w', markerfacecolor='lightgray',
                                 markersize=8, label='Receptors', markeredgecolor='black')]

        for source_type in self.sources_df['source_type'].unique():
            marker = source_markers.get(source_type, 'o')
            legend_elements.append(Line2D([0], [0], marker=marker, color='w',
                                        markerfacecolor='lightgray', markersize=8,
                                        label=f'{source_type.capitalize()} Sources',
                                        markeredgecolor='black', alpha=0.6))

        leg = ax.legend(handles=legend_elements, loc='upper right', frameon=True, fancybox=True, shadow=True,
                 facecolor='white', framealpha=0.95, edgecolor='#BDC3C7', prop={'family': 'Times New Roman', 'size': 11})
        ax.grid(True, alpha=0.3)

        #Set coordinate axis lines to pure black for Figure 14
        for spine in ax.spines.values():
            spine.set_edgecolor('black')
            spine.set_linewidth(1.5)
            spine.set_alpha(1.0)

        plt.tight_layout()
        self._save_fig(fig, 'Figure14_Integrated_Spatial_Map.png')

    def figure15_parameter_sensitivity_analysis(self):

        sensitivity_data = self._get_real_sensitivity_data()

        if not sensitivity_data:
            logging.warning("No parameter sensitivity data available for Figure 15")
            return

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()

        colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']

        for i, (param_name, param_data) in enumerate(sensitivity_data.items()):
            ax = axes[i]

            x_values = param_data['values']
            y_values = param_data['performance']

            # Plot performance curve with markers
            ax.plot(x_values, y_values, 'o-', linewidth=3, markersize=10,
                   color=colors[i], markerfacecolor='white', markeredgecolor='black',
                   markeredgewidth=2, alpha=0.8, label='Performance Score')


            ax.set_xlabel(param_data.get('xlabel', param_name), fontsize=12, fontweight='bold', family='Times New Roman')
            ax.set_ylabel('Model Performance Score', fontsize=12, fontweight='bold', family='Times New Roman')
            ax.set_title(f'Sensitivity: {param_name}', fontsize=14, fontweight='bold', family='Times New Roman', pad=10)
            ax.grid(True, alpha=0.4, linestyle='--')
            ax.tick_params(labelsize=11)

            # Fill area under curve
            ax.fill_between(x_values, y_values, alpha=0.2, color=colors[i])

            # Add legend for each subplot
            ax.legend(loc='best', frameon=True, fancybox=True, shadow=True, framealpha=0.9,
                     prop={'family': 'Times New Roman', 'size': 11})


            # Optimize y-axis limits based on data range or default to 0.5-1.0
            if len(y_values) > 0:
                y_min, y_max = min(y_values), max(y_values)
                y_range = y_max - y_min
                if y_range > 0.1:  # If data has sufficient range, use data-driven limits
                    margin = y_range * 0.1  # 10% margin
                    ax.set_ylim(max(0, y_min - margin), min(1, y_max + margin))
                else:
                    # Default range for small variations
                    ax.set_ylim(0.5, 1.0)
            else:
                # Fallback default range
                ax.set_ylim(0.5, 1.0)

        plt.suptitle('Parameter Sensitivity Analysis: PhysGAT Model Robustness Assessment',
                    fontsize=14, fontweight='bold', y=0.995, family='Times New Roman')
        plt.tight_layout(rect=[0, 0, 1, 0.99])
        self._save_fig(fig, 'Figure15_Parameter_Sensitivity_Analysis.png')

    def _find_closest_index(self, values, target):
        """Find the index closest to the target value."""
        return np.argmin(np.abs(np.array(values) - target))











    def _get_real_sensitivity_data(self):
        """Return parameter sensitivity data if available; otherwise load from JSON/CSV fallback.

        Fallback loader first tries to load sensitivity_results.json (which contains 'current' values),
        then falls back to CSV files if JSON is not available. This ensures that the 'current' parameter
        values are correctly displayed in Figure 15.
        """
        # Primary source: in-memory analysis_results
        if hasattr(self, 'analysis_results') and self.analysis_results:
            sensitivity_results = self.analysis_results.get('parameter_sensitivity', None)
            if sensitivity_results:
                return sensitivity_results

        # Fallback: try to load sensitivity_results.json first, then CSV files
        try:
            import glob, os
            import json

            dirs_to_check = []
            json_candidates = []

            # 1. Check current output directory's sensitivity subdirectory
            if hasattr(self, 'output_dir') and self.output_dir:
                dirs_to_check.append(os.path.join(self.output_dir, 'sensitivity'))

                # 2. Check parent directory's sensitivity subdirectory
                parent_dir = os.path.dirname(self.output_dir)
                if parent_dir:
                    dirs_to_check.append(os.path.join(parent_dir, 'sensitivity'))

                    # 3. Check sensitivity_outputs directory (for independent sensitivity analysis runs)
                    # Get the workspace root (parent of run_outputs)
                    workspace_root = os.path.dirname(parent_dir)
                    sensitivity_outputs_dir = os.path.join(workspace_root, 'sensitivity_outputs')

                    if os.path.isdir(sensitivity_outputs_dir):
                        # Find the most recent sensitivity output directory
                        sensitivity_runs = sorted([d for d in os.listdir(sensitivity_outputs_dir)
                                                  if os.path.isdir(os.path.join(sensitivity_outputs_dir, d))],
                                                 reverse=True)
                        if sensitivity_runs:
                            # Use the most recent run (check parent directory for JSON)
                            most_recent_run_dir = os.path.join(sensitivity_outputs_dir, sensitivity_runs[0])
                            most_recent_sensitivity = os.path.join(most_recent_run_dir, 'sensitivity')
                            dirs_to_check.append(most_recent_sensitivity)

                            # Also check for sensitivity_results.json in the parent directory
                            json_path = os.path.join(most_recent_run_dir, 'sensitivity_results.json')
                            if os.path.isfile(json_path):
                                json_candidates.append(json_path)

                            logging.info(f"Added sensitivity_outputs directory to search: {most_recent_sensitivity}")

            # Try to load from JSON first (preferred, contains 'current' values)
            for json_path in json_candidates:
                try:
                    with open(json_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    if data:
                        logging.info(f"Successfully loaded sensitivity data from JSON: {json_path}")
                        return data
                except Exception as e:
                    logging.warning(f"Failed to load sensitivity JSON {json_path}: {e}")
                    continue

            # Fallback to CSV files if JSON not available
            candidates = []

            for d in dirs_to_check:
                if d and os.path.isdir(d):
                    candidates = glob.glob(os.path.join(d, 'sensitivity_*.csv'))
                    if candidates:
                        logging.info(f"Found {len(candidates)} sensitivity CSV files in: {d}")
                        break

            if not candidates:
                logging.warning("No sensitivity CSV files found in any search directory")
                return None

            data = {}
            for csv_path in candidates:
                try:
                    df = pd.read_csv(csv_path)
                    if 'value' not in df.columns:
                        continue
                    # Prefer provided performance; otherwise derive a monotonic score from final_total
                    if 'performance' in df.columns and not df['performance'].isna().all():
                        y_values = df['performance'].astype(float).tolist()
                    elif 'final_total' in df.columns:
                        # Map lower final_total to higher score in [0,1] using min-max normalization
                        vals = df['final_total'].astype(float).values
                        vmin, vmax = float(np.nanmin(vals)), float(np.nanmax(vals))
                        denom = (vmax - vmin) if (vmax - vmin) > 1e-8 else 1.0
                        y_values = (1.0 - (vals - vmin) / denom).tolist()
                    else:
                        continue

                    param_name = df['param'].iloc[0] if 'param' in df.columns and pd.notna(df['param'].iloc[0]) else os.path.splitext(os.path.basename(csv_path))[0].replace('sensitivity_', '')
                    x_values = df['value'].tolist()

                    # Determine optimal by highest score
                    y_arr = np.array(y_values, dtype=float)
                    opt_idx = int(np.nanargmax(y_arr)) if y_arr.size > 0 else 0
                    optimal_value = float(x_values[opt_idx]) if x_values else None
                    optimal_performance = float(y_arr[opt_idx]) if y_arr.size > 0 else None

                    data[param_name] = {
                        'values': x_values,
                        'performance': y_values,
                        'optimal_value': optimal_value,
                        'optimal_performance': optimal_performance
                    }
                except Exception as e:
                    logging.warning(f"Failed to load sensitivity CSV {csv_path}: {e}")
                    continue

            return data if data else None
        except Exception as e:
            logging.warning(f"Sensitivity data fallback failed: {e}")
            return None
















    def figure7x_loss_weight_evolution_style(self):
        try:
            history = self.model_data.get('training_history', {})
            if not history:
                logging.warning("Training history data not found, skipping Figure 7e generation.")
                return

            required_losses = ['chemical_loss', 'strength_loss', 'distance_loss', 'reconstruction_loss']
            if not all(key in history and history[key] for key in required_losses):
                logging.warning("Required loss data not found, skipping Figure 7e generation.")
                return

            if 'total_loss' not in history or not history['total_loss']:
                total_loss = []
                recon_series = history.get('reconstruction_loss', [0] * len(history['chemical_loss']))
                for i in range(len(history['chemical_loss'])):
                    total = (history['chemical_loss'][i] +
                             history['strength_loss'][i] +
                             history['distance_loss'][i] +
                             recon_series[i])
                    total_loss.append(total)
                history['total_loss'] = total_loss

            fig, ax = plt.subplots(figsize=(12, 8))

            colors = {
                'chemistry': '#3498DB',   # Blue
                'distance':  '#F39C12',   # Orange
                'strength':  '#2ECC71',   # Green
                'reconstruction': '#9B59B6',   # Purple
                'total':     '#E74C3C'    # Red
            }

            epochs = range(1, len(history['chemical_loss']) + 1)

            loss_types = [
                ('chemical_loss', 'chemistry', 'Chemical Loss'),
                ('distance_loss', 'distance', 'Distance Loss'),
                ('strength_loss', 'strength', 'Strength Loss'),
                ('reconstruction_loss', 'reconstruction', 'Reconstruction Loss'),
                ('total_loss', 'total', 'Total Loss')
            ]

            for loss_key, color_key, label in loss_types:
                if loss_key in history and history[loss_key]:
                    values = history[loss_key]
                    line = ax.plot(epochs, values, color=colors[color_key],
                                 linewidth=2.5, label=label, alpha=0.8)

                    ax.fill_between(epochs, 0, values, color=colors[color_key], alpha=0.2)

            ax.set_title('Training Loss Evolution (Enhanced Style)', fontsize=18, fontweight='bold', pad=10, family='Times New Roman')
            ax.set_xlabel('Training Epochs', fontsize=17, fontweight='bold', family='Times New Roman')
            ax.set_ylabel('Loss Value', fontsize=17, fontweight='bold', family='Times New Roman')
            ax.legend(loc='upper right', frameon=True, fancybox=True, shadow=True,
                     prop={'family': 'Times New Roman', 'size': 11})
            ax.grid(True, alpha=0.3)
            ax.set_xlim(1, len(epochs))
            ax.tick_params(labelsize=13)
            ax.set_xticklabels(ax.get_xticklabels(), fontsize=13, family='Times New Roman')
            ax.set_yticklabels(ax.get_yticklabels(), fontsize=13, family='Times New Roman')

            for spine in ax.spines.values():
                spine.set_color('black')
                spine.set_linewidth(1.2)

            plt.tight_layout()
            save_path = os.path.join(self.output_dir, 'Figure7x_Loss_Evolution_Enhanced.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()

            logging.info(f"Figure 7e (enhanced loss evolution) saved: {save_path}")

        except Exception as e:
            logging.error(f"Error generating Figure 7e: {e}")
            import traceback
            traceback.print_exc()

    def figure7f_adaptive_weights_evolution(self):
        """Generate Figure 7f: Adaptive weights evolution during training.

        Shows the real-time changes of 4 loss function weights (chemistry, distance, strength, cosine)
        with constraint that weights sum to 1.0 at all times.
        """
        try:
            history = self.model_data.get('training_history', {})
            if not history:
                logging.warning("Training history data not found, skipping Figure 7f generation.")
                return

            # Check for adaptive weights data
            adaptive_weights = history.get('adaptive_weights', {})
            if not adaptive_weights:
                logging.warning("Adaptive weights data not found, skipping Figure 7f generation.")
                return

            # Use updated key 'reconstruction' instead of legacy 'cosine'
            required_weights = ['chemistry', 'distance', 'strength', 'reconstruction']
            if not all(key in adaptive_weights and adaptive_weights[key] for key in required_weights):
                logging.warning("Required adaptive weights data not found, skipping Figure 7f generation.")
                return

            # Verify all weight series have the same length
            lengths = [len(adaptive_weights[key]) for key in required_weights]
            if len(set(lengths)) > 1:
                logging.warning(f"Inconsistent weight series lengths: {lengths}, skipping Figure 7f generation.")
                return

            if lengths[0] == 0:
                logging.warning("Empty adaptive weights data, skipping Figure 7f generation.")
                return

            # Create the figure with Figure7 series styling
            fig, ax = plt.subplots(figsize=(12, 8))

            # Color scheme consistent with Figure7x
            colors = {
                'chemistry': '#3498DB',   # Blue
                'distance':  '#F39C12',   # Orange
                'strength':  '#2ECC71',   # Green
                'reconstruction': '#9B59B6'    # Purple
            }

            epochs = range(1, lengths[0] + 1)

            # Plot weight evolution curves
            weight_types = [
                ('chemistry', 'Chemistry Loss Weight'),
                ('distance', 'Distance Loss Weight'),
                ('strength', 'Strength Loss Weight'),
                ('reconstruction', 'Reconstruction Loss Weight')
            ]

            for weight_key, label in weight_types:
                values = adaptive_weights[weight_key]
                line = ax.plot(epochs, values, color=colors[weight_key],
                             linewidth=2.5, label=label, alpha=0.8)

                # Add subtle fill area
                ax.fill_between(epochs, 0, values, color=colors[weight_key], alpha=0.15)

            # Verify and show weight sum constraint
            weight_sums = []
            for i in range(lengths[0]):
                total = sum(adaptive_weights[key][i] for key in required_weights)
                weight_sums.append(total)

            # Check if constraint is satisfied (sum ≈ 1.0)
            avg_sum = sum(weight_sums) / len(weight_sums)
            max_deviation = max(abs(s - 1.0) for s in weight_sums)

            # Apply unified styling
            self._apply_unified_legend_style(ax)
            self._apply_unified_border_style(ax)

            ax.set_title('Adaptive Loss Function Weights Evolution', fontsize=18, fontweight='bold', pad=10, family='Times New Roman')
            ax.set_xlabel('Training Epochs', fontsize=17, fontweight='bold', family='Times New Roman')
            ax.set_ylabel('Weight Value', fontsize=17, fontweight='bold', family='Times New Roman')
            ax.grid(True, alpha=0.3)
            ax.set_xlim(1, len(epochs))
            ax.tick_params(labelsize=13)
            ax.set_xticklabels(ax.get_xticklabels(), fontsize=13, family='Times New Roman')
            ax.set_yticklabels(ax.get_yticklabels(), fontsize=13, family='Times New Roman')
            # Set fixed Y-axis range for Figure 7f to [0.0, 0.5]
            ax.set_ylim(0.0, 0.5)

            # Add learning rate curve on right Y-axis
            if 'learning_rate' in history and history['learning_rate']:
                ax2 = ax.twinx()
                lr_data = history['learning_rate']
                # Ensure learning rate data matches epochs length
                if len(lr_data) >= len(epochs):
                    lr_epochs = epochs
                    lr_values = lr_data[:len(epochs)]
                else:
                    lr_epochs = range(1, len(lr_data) + 1)
                    lr_values = lr_data

                ax2.plot(lr_epochs, lr_values, color='#E74C3C', linewidth=2,
                        linestyle='--', label='Learning Rate', alpha=0.8)
                ax2.set_ylabel('Learning Rate', fontsize=14, family='Times New Roman')
                ax2.tick_params(axis='y', labelcolor='#E74C3C', labelsize=13)
                ax2.set_yticklabels(ax2.get_yticklabels(), fontsize=13, family='Times New Roman')

                # Apply consistent styling to right axis
                for spine in ax2.spines.values():
                    spine.set_edgecolor('black')
                    spine.set_alpha(1.0)

            # Add constraint verification text
            constraint_text = f'Weight Sum: {avg_sum:.4f} ± {max_deviation:.4f}'
            constraint_color = 'green' if max_deviation < 0.01 else 'orange'
            ax.text(0.02, 0.98, constraint_text, transform=ax.transAxes,
                   fontsize=10, verticalalignment='top', family='Times New Roman',
                   bbox=dict(boxstyle='round', facecolor=constraint_color, alpha=0.3))

            plt.tight_layout()
            save_path = os.path.join(self.output_dir, 'Figure7f_Adaptive_Weights_Evolution.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()

            logging.info(f"Figure 7f (adaptive weights evolution) saved: {save_path}")
            logging.info(f"Weight constraint verification - Average sum: {avg_sum:.6f}, Max deviation: {max_deviation:.6f}")

        except Exception as e:
            logging.error(f"Error generating Figure 7f: {e}")
            import traceback
            traceback.print_exc()

    def figure16_dem_surface_3d(self):
        """Generate Figure 16: DEM 3D terrain surface with contours (surfc style)"""
        try:
            # Parse DEM path - handle both dict and DictConfig objects
            dem_path = None

            # Try multiple ways to access dem_path from config
            if isinstance(self.config, dict):
                dem_path = self.config.get('dem_path')
            else:
                # Handle DictConfig or other config objects
                try:
                    dem_path = self.config.dem_path if hasattr(self.config, 'dem_path') else None
                except:
                    dem_path = None

            # If still not found, try graph.dem_path
            if dem_path is None:
                try:
                    if hasattr(self.config, 'graph') and hasattr(self.config.graph, 'dem_path'):
                        dem_path = self.config.graph.dem_path
                except:
                    pass

            if dem_path is None or not os.path.exists(dem_path):
                logging.warning(f"DEM path does not exist or is None: {dem_path}, skipping Figure 16 generation")
                return

            import rasterio
            from rasterio.windows import Window
            from rasterio import transform as rio_transform

            # Calculate spatial extent of receptors and sources for efficient clipping
            xs = np.concatenate([self.receptors_df['lon'].values, self.sources_df['lon'].values])
            ys = np.concatenate([self.receptors_df['lat'].values, self.sources_df['lat'].values])
            minx, maxx = np.nanmin(xs), np.nanmax(xs)
            miny, maxy = np.nanmin(ys), np.nanmax(ys)

            with rasterio.open(dem_path) as src:
                # Get pixel coordinates for the bounding box
                r0, c0 = src.index(minx, maxy)
                r1, c1 = src.index(maxx, miny)
                rmin, rmax = sorted([r0, r1])
                cmin, cmax = sorted([c0, c1])

                # Add padding for better visualization
                pad = 50
                rmin = max(rmin - pad, 0)
                cmin = max(cmin - pad, 0)
                rmax = min(rmax + pad, src.height - 1)
                cmax = min(cmax + pad, src.width - 1)

                # Read DEM data within the window
                height = max(1, rmax - rmin)
                width = max(1, cmax - cmin)
                window = Window(cmin, rmin, width, height)
                Z = src.read(1, window=window)
                Z = np.where(np.isfinite(Z), Z, np.nan)

                # Create coordinate meshgrid
                transform = src.window_transform(window)
                cols = np.arange(Z.shape[1])
                rows = np.arange(Z.shape[0])
                X, Y = np.meshgrid(cols, rows)

                # Convert pixel coordinates to geographic coordinates
                lon_coords = transform[2] + X * transform[0]
                lat_coords = transform[5] + Y * transform[4]

            # Remove NaN values for better visualization
            valid_mask = ~np.isnan(Z)
            if not np.any(valid_mask):
                logging.warning("No valid elevation data found, skipping Figure 16")
                return

            # Ultra-high resolution - further reduce downsampling for finest topographic detail
            step = max(1, int(max(Z.shape) / 600))  # Ultra-high resolution (increased from 400 to 600)
            if step > 1:
                Xs = lon_coords[::step, ::step]
                Ys = lat_coords[::step, ::step]
                Zs = Z[::step, ::step]
            else:
                Xs, Ys, Zs = lon_coords, lat_coords, Z

            # Create 3D plot with enhanced settings for high resolution
            from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
            fig = plt.figure(figsize=(16, 12))  # Increased figure size for better resolution
            ax = fig.add_subplot(111, projection='3d')

            # Enhanced lighting and shading with realistic settings
            ls = LightSource(azdeg=315, altdeg=45)

            # Filter out zero values (water bodies or no-data areas) for realistic elevation range
            valid_mask = Zs > 0
            if np.any(valid_mask):
                valid_elevations = Zs[valid_mask]
                zmin_valid = np.min(valid_elevations)
                zmax_valid = np.max(valid_elevations)
                elevation_range_valid = zmax_valid - zmin_valid

                # Use valid data range for statistics
                logging.info(f"DEM terrain statistics (excluding zero values):")
                logging.info(f"  Valid elevation range: {zmin_valid:.1f} - {zmax_valid:.1f} m")
                logging.info(f"  Valid elevation difference: {elevation_range_valid:.1f} m")
                logging.info(f"  Mean valid elevation: {np.mean(valid_elevations):.1f} m")
                logging.info(f"  Zero-value points (water/no-data): {np.sum(~valid_mask)}")

                # Use valid range for color mapping, but keep original data for surface
                zmin_for_color = zmin_valid
                zmax_for_color = zmax_valid
            else:
                # Fallback to original range if no valid data
                zmin_for_color = np.nanmin(Zs)
                zmax_for_color = np.nanmax(Zs)
                elevation_range_valid = zmax_for_color - zmin_for_color
                logging.warning("No valid elevation data found, using full range including zeros")

            # Normalize elevation for better color mapping (using valid range)
            Zn = np.clip((Zs - zmin_for_color) / (zmax_for_color - zmin_for_color + 1e-8), 0, 1)

            # Use no vertical exaggeration for realistic representation
            vert_exag = 1.0
            shaded = ls.shade(Zn, cmap=plt.cm.terrain, vert_exag=vert_exag, blend_mode='soft')
            logging.info(f"  Using vertical exaggeration: {vert_exag}x (realistic scale)")

            # Elevate the 3D surface dramatically above the base contours for extreme visual separation
            elevation_offset = elevation_range_valid * 22.5  # Lift surface by 2250% of elevation range (~70km separation)
            Zs_elevated = Zs + elevation_offset

            # Plot 3D surface with realistic parameters (elevated)
            surf = ax.plot_surface(Xs, Ys, Zs_elevated,
                                 rstride=max(1, step//2), cstride=max(1, step//2),
                                 facecolors=shaded,
                                 linewidth=0.1,
                                 antialiased=True,
                                 shade=True,
                                 alpha=0.9)

            # Add contour lines at the base with appropriate spacing
            num_contours = min(15, max(8, int(elevation_range_valid / 200)))  # Based on valid elevation range
            contour_levels = np.linspace(zmin_for_color, zmax_for_color, num_contours)

            # Calculate plot space boundaries for proper positioning
            z_plot_min = np.nanmin(Zs) - elevation_range_valid * 0.5  # Bottom of plot space
            z_plot_max = np.nanmax(Zs) + elevation_range_valid * 0.3   # Top of plot space

            # Position 2D contour lines at the absolute bottom of the coordinate system
            # Only keep colored elevation contours for clear topographic information
            base_offset = z_plot_min
            ax.contour(Xs, Ys, Zs, zdir='z', offset=base_offset,
                      cmap='terrain', levels=contour_levels,
                      linewidths=0.8, alpha=0.8)

            # Remove gray/black surface contour lines to eliminate visual clutter
            # Only the colored bottom contours are retained for clear topographic reference

            # Enhanced styling - remove Z-axis labels since colors represent elevation
            ax.set_xlabel('Longitude (°)', fontsize=12, fontweight='bold', family='Times New Roman')
            ax.set_ylabel('Latitude (°)', fontsize=12, fontweight='bold', family='Times New Roman')
            ax.set_zlabel('')  # Remove Z-axis label to reduce visual clutter
            ax.set_zticks([])  # Remove Z-axis tick marks and labels
            ax.set_title('3D DEM Terrain Surface',
                        fontsize=24, fontweight='bold', pad=0, family='Times New Roman', loc='center')
            ax.tick_params(labelsize=11)

            # Use realistic viewing angle
            ax.view_init(elev=25, azim=-45)

            # Calculate realistic aspect ratio based on actual geographic distances
            lon_range = np.nanmax(Xs) - np.nanmin(Xs)  # degrees
            lat_range = np.nanmax(Ys) - np.nanmin(Ys)  # degrees

            # Convert to kilometers for realistic proportions
            lat_center = (np.nanmax(Ys) + np.nanmin(Ys)) / 2
            lon_km = lon_range * 111 * np.cos(np.radians(lat_center))  # longitude to km
            lat_km = lat_range * 111  # latitude to km
            elev_km = elevation_range_valid / 1000  # elevation to km

            # Calculate enhanced Z-axis ratio with vertical exaggeration for better terrain visibility
            horizontal_scale = max(lon_km, lat_km)
            base_aspect_z = elev_km / horizontal_scale

            # Apply enhanced vertical exaggeration (10x) for optimal mountainous terrain representation
            vertical_exaggeration = 10.0
            enhanced_aspect_z = base_aspect_z * vertical_exaggeration

            # Apply the enhanced aspect ratio for better terrain representation
            ax.set_box_aspect([1, 1, enhanced_aspect_z])

            # Set Z-axis limits to accommodate both contours at bottom and elevated surface
            ax.set_zlim(z_plot_min, np.nanmax(Zs_elevated) + elevation_range_valid * 0.1)

            logging.info(f"  Geographic distances: {lon_km:.1f}km (E-W) × {lat_km:.1f}km (N-S) × {elev_km:.1f}km (elevation)")
            logging.info(f"  Base Z-axis aspect ratio: {base_aspect_z:.3f}")
            logging.info(f"  Enhanced Z-axis aspect ratio (10x exaggeration): {enhanced_aspect_z:.3f}")
            logging.info(f"  Vertical separation: ~{elevation_offset/1000:.1f}km between surface and contours")

            # Add colorbar for elevation reference
            mappable = plt.cm.ScalarMappable(cmap='terrain')
            mappable.set_array(Zs)
            cbar = fig.colorbar(mappable, ax=ax, shrink=0.65, aspect=15, pad=0.05)
            cbar.set_label('Elevation (m)', fontsize=12, fontweight='bold', family='Times New Roman')
            cbar.ax.tick_params(labelsize=11)
            for label in cbar.ax.get_yticklabels():
                label.set_family('Times New Roman')

            # Improve layout with adjusted spacing for title and bottom margin
            fig.tight_layout(pad=1.5, rect=[0, 0.1, 1, 0.98])

            # Save with ultra-high DPI for publication quality
            self._save_fig_high_res(fig, 'Figure16_DEM_Surface_3D.png')

            logging.info("Figure 16 3D DEM surface generated successfully")

        except Exception as e:
            logging.error(f"Failed to generate Figure 16 (DEM surface): {e}")
            import traceback; traceback.print_exc()

            traceback.print_exc()


