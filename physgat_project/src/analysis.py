#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analysis module for post-processing, report generation, and pollution index calculation.
Includes Nemerow index calculation, contribution reports, and GNNExplainer analysis.

分析模块，用于后处理、报告生成和污染指数计算。
包含内梅罗指数计算、贡献度报告生成和GNNExplainer可解释性分析。

Author: Wenhao Wang
"""
import pandas as pd
import numpy as np
import torch
import logging

# ===================================================================
#  Primary Analysis Functions
# ===================================================================

def calculate_nemerow_index(receptors_df: pd.DataFrame, metals: list, config=None) -> pd.DataFrame:

    logging.info("=== Calculating Nemerow Pollution Index ===")

    if config and hasattr(config, 'pollution_analysis') and hasattr(config.pollution_analysis, 'metal_weights'):
        metal_weights = dict(config.pollution_analysis.metal_weights)
    else:
        metal_weights = {'Hg': 3, 'Pb': 3, 'Cd': 3, 'As': 3, 'Zn': 2, 'Cu': 2, 'Cr': 2, 'Ni': 2}

    if config and hasattr(config, 'pollution_analysis') and hasattr(config.pollution_analysis, 'pollution_levels'):
        levels = config.pollution_analysis.pollution_levels
        clean_threshold = levels.clean
        still_clean_threshold = levels.still_clean
        lightly_polluted_threshold = levels.lightly_polluted
        moderately_polluted_threshold = levels.moderately_polluted
    else:
        clean_threshold = 0.7
        still_clean_threshold = 1.0
        lightly_polluted_threshold = 2.0
        moderately_polluted_threshold = 3.0

    def calculate_nemerow_for_receptor(receptor_data):
        pollution_indices, weights = [], []
        for metal in metals:
            background_col = 'bg_' + metal
            if metal in receptor_data and background_col in receptor_data:
                receptor_value = receptor_data[metal]
                background_value = receptor_data[background_col]
                if pd.notna(receptor_value) and pd.notna(background_value) and background_value > 0:
                    net_pollution = receptor_value - background_value

                    if net_pollution <= 0:
                        pollution_index = 0.0
                    else:
                        pollution_index = receptor_value / background_value

                    pollution_indices.append(pollution_index)
                    weights.append(metal_weights.get(metal, 1))

        if pollution_indices and any(pi > 0 for pi in pollution_indices):
            pi_max = max(pollution_indices)
            pi_avg_weighted = np.average(pollution_indices, weights=weights)
            nemerow_index = np.sqrt((pi_max**2 + pi_avg_weighted**2) / 2)

            if nemerow_index <= clean_threshold:
                level = "Clean"
            elif clean_threshold < nemerow_index <= still_clean_threshold:
                level = "Still Clean"
            elif still_clean_threshold < nemerow_index <= lightly_polluted_threshold:
                level = "Lightly Polluted"
            elif lightly_polluted_threshold < nemerow_index <= moderately_polluted_threshold:
                level = "Moderately Polluted"
            else:
                level = "Heavily Polluted"

            return nemerow_index, level
        else:
            return 0.0, "Clean"

    nemerow_results = receptors_df.apply(calculate_nemerow_for_receptor, axis=1, result_type='expand')
    receptors_df['nemerow_index'] = nemerow_results[0]
    receptors_df['pollution_level'] = nemerow_results[1]
    logging.info("Nemerow index calculation complete.")
    return receptors_df

def get_source_contributions_summary(predictions_df: pd.DataFrame, sources_df: pd.DataFrame):
    """Gets a summary of source contributions."""
    logging.info("=== Generating contribution summary ===")
    type_summary = predictions_df.groupby(['receptor_idx', 'receptor_sample', 'source_type']).agg(contribution_percent=('contribution_percent_total', 'sum')).reset_index()
    pivot_summary = pd.pivot_table(type_summary, index=['receptor_idx', 'receptor_sample'], columns='source_type', values='contribution_percent').fillna(0)
    return type_summary, pivot_summary

def get_normalized_source_contributions_summary(top5_report_df: pd.DataFrame):
    """Gets a summary of all source contributions with normalized percentages (100% per receptor)."""
    logging.info("=== Generating all source contribution summary ===")

    if top5_report_df.empty:
        logging.warning("All source report is empty")
        return None, None

    type_summary = top5_report_df.groupby(['receptor_idx', 'receptor_sample', 'source_type']).agg(
        contribution_percent=('contribution_percent', 'sum')
    ).reset_index()

    pivot_summary = pd.pivot_table(
        type_summary,
        index=['receptor_idx', 'receptor_sample'],
        columns='source_type',
        values='contribution_percent'
    ).fillna(0)

    row_sums = pivot_summary.sum(axis=1)
    for idx, total in row_sums.items():
        if abs(total - 100.0) > 0.1:  
            logging.warning(f"Receptor {idx}: contributions sum to {total:.2f}% (expected 100%)")

    return type_summary, pivot_summary



def _validate_indices(predictions_df: pd.DataFrame, receptors_df: pd.DataFrame, sources_df: pd.DataFrame) -> pd.DataFrame:
    logging.info("Validating prediction indices...")

    initial_count = len(predictions_df)
    max_receptor_idx = len(receptors_df) - 1
    max_source_idx = len(sources_df) - 1

    valid_mask = (
        (predictions_df['receptor_idx'] >= 0) &
        (predictions_df['receptor_idx'] <= max_receptor_idx) &
        (predictions_df['source_idx'] >= 0) &
        (predictions_df['source_idx'] <= max_source_idx)
    )

    cleaned_df = predictions_df[valid_mask].copy()
    removed_count = initial_count - len(cleaned_df)

    if removed_count > 0:
        logging.warning(f"Removed {removed_count} predictions with invalid indices:")
        logging.warning(f"  Valid receptor range: [0, {max_receptor_idx}]")
        logging.warning(f"  Valid source range: [0, {max_source_idx}]")

        invalid_df = predictions_df[~valid_mask]
        if len(invalid_df) > 0:
            invalid_receptors = invalid_df[
                (invalid_df['receptor_idx'] < 0) | (invalid_df['receptor_idx'] > max_receptor_idx)
            ]['receptor_idx'].unique()
            invalid_sources = invalid_df[
                (invalid_df['source_idx'] < 0) | (invalid_df['source_idx'] > max_source_idx)
            ]['source_idx'].unique()

            if len(invalid_receptors) > 0:
                logging.warning(f"  Invalid receptor indices: {invalid_receptors[:10]}...")
            if len(invalid_sources) > 0:
                logging.warning(f"  Invalid source indices: {invalid_sources[:10]}...")

    logging.info(f"Index validation complete: {len(cleaned_df)}/{initial_count} predictions retained")
    return cleaned_df



def _load_config_if_needed(config):
    if config is None:
        from omegaconf import OmegaConf
        import os
        current_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(current_dir, "..", "configs", "default.yaml")
        config = OmegaConf.load(config_path)
    return config

def _process_receptor_contributions(receptor_idx, receptor_data, receptors_df, sources_df):
    from .data_utils import _calculate_distance

    receptor_info = receptors_df.iloc[receptor_idx]
    receptor_coord = (receptor_info['lat'], receptor_info['lon'])

    if receptor_data.empty:
        logging.info(f"Receptor {receptor_idx} has no available sources")
        return []

    valid_df = receptor_data.copy()

    all_sources = valid_df.sort_values('contribution_percent_total', ascending=False).copy()

    total = all_sources['contribution_percent_total'].sum()
    if total > 0:
        all_sources['contribution_percent_total'] = (all_sources['contribution_percent_total'] / total * 100).round(2)
        normalized_total = 100.0
    else:
        n_sources = len(all_sources)
        all_sources['contribution_percent_total'] = 100.0 / n_sources if n_sources > 0 else 0.0
        normalized_total = 100.0

    all_sources['rank'] = np.arange(1, len(all_sources) + 1)
    all_sources['contribution_display'] = all_sources['contribution_percent_total'].map(lambda v: f"{v:.2f}%") + \
                                           " (Rank " + all_sources['rank'].astype(str) + ")"

    logging.info(f"Receptor {receptor_idx}: All source contribution total = {normalized_total:.2f}%, number of sources = {len(all_sources)}")

    for idx, v in enumerate(all_sources['contribution_percent_total'].head(10).tolist(), start=1):
        logging.info(f"  Rank {idx}: {v:.2f}%")

    return all_sources

def _create_result_records(all_sources, receptors_df, sources_df):
    from .data_utils import _calculate_distance

    if len(all_sources) == 0:
        return []

    rec_map = receptors_df.reset_index().rename(columns={'index': 'receptor_idx', 'sample': 'receptor_sample'})
    src_map = sources_df.reset_index().rename(columns={'index': 'source_idx', 'sample': 'source_sample'})

    merged = (all_sources
              .merge(rec_map[['receptor_idx', 'receptor_sample', 'lon', 'lat']], on='receptor_idx', how='left')
              .rename(columns={'lon': 'receptor_lon', 'lat': 'receptor_lat'})
              .merge(src_map[['source_idx', 'source_sample', 'source_type', 'lon', 'lat']], on='source_idx', how='left', suffixes=('', '_src'))
              .rename(columns={'lon': 'source_lon', 'lat': 'source_lat'}))
    if 'receptor_sample_x' in merged.columns or 'receptor_sample_y' in merged.columns:
        merged['receptor_sample'] = (
            (merged['receptor_sample'] if 'receptor_sample' in merged.columns else pd.Series(index=merged.index, dtype=object))
            .fillna(merged['receptor_sample_x'] if 'receptor_sample_x' in merged.columns else None)
            .fillna(merged['receptor_sample_y'] if 'receptor_sample_y' in merged.columns else None)
        )
        merged.drop(columns=[c for c in ['receptor_sample_x', 'receptor_sample_y'] if c in merged.columns], inplace=True)

    if 'source_sample_src' in merged.columns:
        if 'source_sample' not in merged.columns:
            merged['source_sample'] = merged['source_sample_src']
        else:
            merged['source_sample'] = merged['source_sample'].fillna(merged['source_sample_src'])
        merged.drop(columns=['source_sample_src'], inplace=True)

    if 'source_type_src' in merged.columns:
        if 'source_type' not in merged.columns:
            merged['source_type'] = merged['source_type_src']
        else:
            merged['source_type'] = merged['source_type'].fillna(merged['source_type_src'])
        merged.drop(columns=['source_type_src'], inplace=True)


    if 'contribution_percent_std' in merged.columns:
        merged['contribution_std'] = merged['contribution_percent_std']
    elif 'contribution_std' not in merged.columns:
        merged['contribution_std'] = 0.0

    if 'source_combined_name' in merged.columns:
        merged['source_sample'] = np.where(merged['source_combined_name'].notna(), merged['source_combined_name'], merged['source_sample'])

    merged['distance_km'] = merged.apply(
        lambda r: _calculate_distance(r['receptor_lat'], r['receptor_lon'], r['source_lat'], r['source_lon']), axis=1
    )

    merged['is_combined_source'] = False

    result_cols = [
        'receptor_idx', 'receptor_sample', 'receptor_lon', 'receptor_lat',
        'source_idx', 'source_sample', 'source_type', 'source_lon', 'source_lat',
        'contribution_percent_total', 'contribution_std', 'distance_km', 'is_combined_source'
    ]

    merged = merged[result_cols].rename(columns={'contribution_percent_total': 'contribution_percent'})

    return merged.to_dict(orient='records')

def generate_contribution_report(predictions_df: pd.DataFrame, receptors_df: pd.DataFrame, sources_df: pd.DataFrame, config=None):
    logging.info("=" * 60)
    logging.info("GENERATING COMPREHENSIVE SOURCE CONTRIBUTION REPORT")
    logging.info("Ensuring 100% normalization for all receptors")
    logging.info("=" * 60)

    predictions_df = _validate_indices(predictions_df, receptors_df, sources_df)

    if predictions_df.empty:
        logging.error("No valid predictions after index validation")
        return pd.DataFrame()

    config = _load_config_if_needed(config)
    logging.info("Removing candidate selection constraints, including all source contributions")

    all_results = []

    for receptor_idx in predictions_df['receptor_idx'].unique():
        receptor_data = predictions_df[predictions_df['receptor_idx'] == receptor_idx]

        all_sources = _process_receptor_contributions(receptor_idx, receptor_data, receptors_df, sources_df)

        if len(all_sources) > 0:
            receptor_results = _create_result_records(all_sources, receptors_df, sources_df)
            all_results.extend(receptor_results)

    return pd.DataFrame(all_results)

def calculate_primary_secondary_pollutants(receptors_df: pd.DataFrame, metals: list, config=None) -> pd.DataFrame:
    logging.info("=== Calculating primary and secondary pollutants ===")

    if config and hasattr(config, 'pollution_analysis') and hasattr(config.pollution_analysis, 'primary_secondary_weights'):
        metal_weights = dict(config.pollution_analysis.primary_secondary_weights)
    else:
        metal_weights = {
            'Cr': 3,   
            'Ni': 2,   
            'Cu': 2,   
            'Zn': 1,   
            'As': 3,   
            'Cd': 3,   
            'Pb': 3    
        }

    def calculate_pollutants_for_receptor(row):
        idx = row.name
        receptor_sample = row.get('receptor_sample', row.get('sample', f'Receptor_{idx}'))

        net_pollution = {}
        weighted_net_pollution = {}

        for metal in metals:
            if metal in row:
                bg_col = f'bg_{metal}'
                if bg_col in row and pd.notna(row[bg_col]) and row[bg_col] > 0:
                    net_value = max(0, row[metal] - row[bg_col])  
                    net_pollution[metal] = net_value

                    weight = metal_weights.get(metal, 1)
                    weighted_net_pollution[metal] = net_value * weight
                else:
                    net_pollution[metal] = max(0, row[metal])
                    weight = metal_weights.get(metal, 1)
                    weighted_net_pollution[metal] = row[metal] * weight

        sorted_pollutants = sorted(weighted_net_pollution.items(),
                                 key=lambda x: x[1], reverse=True)

        primary_pollutant = sorted_pollutants[0][0] if sorted_pollutants else 'None'
        secondary_pollutant = sorted_pollutants[1][0] if len(sorted_pollutants) > 1 else 'None'

        primary_net_value = net_pollution.get(primary_pollutant, 0)
        secondary_net_value = net_pollution.get(secondary_pollutant, 0)

        primary_bg_col = f'bg_{primary_pollutant}'
        secondary_bg_col = f'bg_{secondary_pollutant}'

        primary_pollution_index = 0
        secondary_pollution_index = 0

        if primary_bg_col in row and row[primary_bg_col] > 0:
            primary_pollution_index = row[primary_pollutant] / row[primary_bg_col]

        if secondary_bg_col in row and row[secondary_bg_col] > 0:
            secondary_pollution_index = row[secondary_pollutant] / row[secondary_bg_col]

        def get_pollution_level(pollution_index):
            if pollution_index <= 1.0:
                return "No pollution"
            elif pollution_index <= 2.0:
                return "Light pollution"
            elif pollution_index <= 3.0:
                return "Moderate pollution"
            elif pollution_index <= 5.0:
                return "Heavy pollution"
            else:
                return "Severe pollution"

        primary_level = get_pollution_level(primary_pollution_index)
        secondary_level = get_pollution_level(secondary_pollution_index)

        return pd.Series({
            'receptor_sample': receptor_sample,
            'receptor_idx': idx,
            'primary_pollutant': primary_pollutant,
            'primary_net_value': round(primary_net_value, 3),
            'primary_pollution_index': round(primary_pollution_index, 3),
            'primary_pollution_level': primary_level,
            'secondary_pollutant': secondary_pollutant,
            'secondary_net_value': round(secondary_net_value, 3),
            'secondary_pollution_index': round(secondary_pollution_index, 3),
            'secondary_pollution_level': secondary_level,
            'total_weighted_pollution': round(sum(weighted_net_pollution.values()), 3),
            'nemerow_index': row.get('nemerow_index', 0)
        })

    result_df = receptors_df.apply(calculate_pollutants_for_receptor, axis=1)
    logging.info(f"Primary/secondary pollutant calculation completed, processed {len(result_df)} receptors")

    return result_df

def filter_and_renormalize_contributions(predictions_df: pd.DataFrame, receptors_df: pd.DataFrame, sources_df: pd.DataFrame, config=None):

    logging.info("Processing contribution rate data with value-weighted correction...")

    original_contributions = predictions_df['contribution'].copy()
    logging.info(f"Original contribution statistics:")
    logging.info(f"  - Data points: {len(original_contributions)}")
    logging.info(f"  - Mean: {original_contributions.mean():.6f}")
    logging.info(f"  - Std: {original_contributions.std():.6f}")
    logging.info(f"  - Range: [{original_contributions.min():.6f}, {original_contributions.max():.6f}]")


    predictions_df['contribution'] = predictions_df['contribution'].clip(lower=0)

    metals = getattr(config, 'metals', None) if config is not None else None
    if metals is None:
        metals = [c for c in sources_df.columns if c.lower() in ['as','cd','cr','cu','hg','pb','zn']]

    #    w_r = softmax(receptor_metals * 5.0)
    receptor_weights = {}
    for ridx in predictions_df['receptor_idx'].unique():
        rrow = receptors_df.iloc[int(ridx)]
        r_vals = np.array([max(rrow.get(m, 0.0), 0.0) for m in metals], dtype=float)
        if np.all(np.isnan(r_vals)):
            r_vals = np.zeros_like(r_vals)
        # softmax with temperature 5.0
        exps = np.exp((r_vals - np.nanmean(r_vals)) * 5.0)
        w = exps / (np.sum(exps) + 1e-8)
        receptor_weights[int(ridx)] = w

    logging.info("Performing value-weighted normalization by receptor...")
    normalized_contributions = []
    for receptor_idx in predictions_df['receptor_idx'].unique():
        receptor_mask = predictions_df['receptor_idx'] == receptor_idx
        receptor_data = predictions_df[receptor_mask].copy()

        w_r = receptor_weights[int(receptor_idx)]

        src_indices = receptor_data['source_idx'].astype(int).values
        s_vals_mat = sources_df.loc[src_indices, metals].clip(lower=0).fillna(0.0).to_numpy(dtype=float)
        v_s_arr = (s_vals_mat * w_r.reshape(1, -1)).sum(axis=1)

        contributions = receptor_data['contribution'].values.astype(float)
        eff = contributions * v_s_arr

        total_eff = eff.sum()

        if total_eff > 1e-12:  
            normalized_contribs_percent = (eff / total_eff) * 100.0
        else:
            n_sources = len(eff)
            normalized_contribs_percent = np.full(n_sources, 100.0 / n_sources) if n_sources > 0 else np.array([])
            logging.warning(f"Receptor {receptor_idx} has zero total effective contribution, using equal distribution")

        receptor_data['contribution_percent_total'] = normalized_contribs_percent
        normalized_contributions.append(receptor_data)

    predictions_df = pd.concat(normalized_contributions, ignore_index=True)

    processed_contributions = predictions_df['contribution_percent_total']
    logging.info(f"Value-weighted contribution rate statistics:")
    logging.info(f"  - Mean: {processed_contributions.mean():.2f}%")
    logging.info(f"  - Std: {processed_contributions.std():.2f}%")
    logging.info(f"  - Range: [{processed_contributions.min():.2f}%, {processed_contributions.max():.2f}%]")

    total_contributions_percent = predictions_df.groupby('receptor_idx')['contribution_percent_total'].sum()
    avg_total = total_contributions_percent.mean()
    std_total = total_contributions_percent.std()

    logging.info(f"Normalization validation - Mean total: {avg_total:.2f}%, Std: {std_total:.2f}%")
    logging.info(f"Total range: [{total_contributions_percent.min():.2f}%, {total_contributions_percent.max():.2f}%]")

    return predictions_df


def run_full_analysis(predictions_df: pd.DataFrame, dataset, config: dict):
    """Runs the full post-processing analysis workflow."""
    logging.info("=== Starting full post-processing analysis workflow ===")
    receptors_df, sources_df = dataset.receptors_df.copy(), dataset.sources_df.copy()

    if 'contribution' in predictions_df.columns:
        predictions_df = filter_and_renormalize_contributions(predictions_df, receptors_df, sources_df, config)

    pred_df_full = predictions_df.merge(
        receptors_df[['sample']].reset_index().rename(columns={'index': 'receptor_idx', 'sample': 'receptor_sample'}), on='receptor_idx'
    ).merge(
        sources_df[['sample', 'source_type']].reset_index().rename(columns={'index': 'source_idx', 'sample': 'source_sample'}), on='source_idx'
    )


    if 'contribution_percent_mean' in pred_df_full.columns:
        pred_df_full.rename(columns={'contribution_percent_mean': 'contribution_percent_total'}, inplace=True)

    backgrounds_df = dataset.unified_backgrounds_df.copy()
    backgrounds_df.rename(columns=lambda c: 'bg_' + c, inplace=True)
    receptors_df_with_bg = pd.concat([receptors_df.reset_index(drop=True), backgrounds_df.reset_index(drop=True)], axis=1)
    receptors_df_analyzed = calculate_nemerow_index(receptors_df_with_bg, config.metals, config)
    receptors_df_analyzed.rename(columns={'sample': 'receptor_sample'}, inplace=True)

    type_summary, pivot_summary = get_source_contributions_summary(pred_df_full, sources_df)

    cols_to_merge = ['receptor_sample', 'nemerow_index', 'pollution_level']
    pivot_summary_reset = pivot_summary.reset_index()
    merged_pivot = pd.merge(pivot_summary_reset, receptors_df_analyzed[cols_to_merge], on='receptor_sample', how='left')

    source_cols = list(pivot_summary.columns)
    fertilizer_index = source_cols.index('fertilizer') + 1 if 'fertilizer' in source_cols else len(source_cols)
    final_pivot_cols = source_cols[:fertilizer_index] + ['nemerow_index', 'pollution_level'] + source_cols[fertilizer_index:]
    final_pivot_cols = ['receptor_idx', 'receptor_sample'] + [c for c in final_pivot_cols if c in merged_pivot.columns]
    final_pivot = merged_pivot[final_pivot_cols].sort_values(by='receptor_idx').reset_index(drop=True)

    try:
        logging.info("Generating missing type fill candidate list (manure disabled)...")
        metals = getattr(config, 'metals', ['As','Cd','Cr','Cu','Hg','Pb','Zn'])
        type_id_to_name = {0: 'atmosphere', 1: 'irrigation', 2: 'fertilizer', 3: 'manure'}
        valid_types_for_fill = ['atmosphere', 'irrigation', 'fertilizer']

        receptor_weights = {}
        for ridx in receptors_df.index:
            rrow = receptors_df.iloc[int(ridx)]
            r_vals = np.array([max(rrow.get(m, 0.0), 0.0) for m in metals], dtype=float)
            exps = np.exp((r_vals - np.nanmean(r_vals)) * 5.0)
            w = exps / (np.sum(exps) + 1e-8)
            receptor_weights[int(ridx)] = w

        existing_by_receptor = pred_df_full.groupby('receptor_idx')['source_type'].apply(lambda s: set(s.tolist())).to_dict()

        fallback_rows = []
        for ridx in receptors_df.index:
            have_types = existing_by_receptor.get(int(ridx), set())
            need_types = [t for t in valid_types_for_fill if t not in have_types]
            if not need_types:
                continue
            w_r = receptor_weights[int(ridx)]
            for tname in need_types:
                cand = sources_df[sources_df['source_type'] == tname].reset_index().rename(columns={'index': 'source_idx'})
                if cand.empty:
                    continue
                s_vals = cand[metals].clip(lower=0).fillna(0.0).to_numpy(dtype=float)
                scores = (s_vals * w_r.reshape(1, -1)).sum(axis=1)
                top_i = int(np.nanargmax(scores))
                fallback_rows.append({
                    'receptor_idx': int(ridx),
                    'receptor_sample': receptors_df.iloc[int(ridx)]['sample'],
                    'source_idx': int(cand.iloc[top_i]['source_idx']),
                    'source_sample': cand.iloc[top_i]['sample'],
                    'source_type': tname,
                    'score_value': float(scores[top_i]),
                    'reason': 'no_candidate_extra_top1'
                })
        fallback_df = pd.DataFrame(fallback_rows)
        if not fallback_df.empty:
            logging.info(f"Fallback candidate generation completed: {len(fallback_df)} entries (for reference only, no effect on training/normalization)")
        else:
            logging.info("No fallback candidates needed")
    except Exception as e:
        logging.warning(f"Failed to generate fallback candidate list: {e}")
        fallback_df = pd.DataFrame()

    top5_report_df = generate_contribution_report(pred_df_full, receptors_df, sources_df, config)

    top5_type_summary, top5_pivot_summary = get_normalized_source_contributions_summary(top5_report_df)

    primary_secondary_pollutants = calculate_primary_secondary_pollutants(receptors_df_analyzed, config.metals, config)

    analysis_results = {
        "receptors_with_nemerow": receptors_df_analyzed,
        "contribution_pivot_summary": final_pivot,  
        "top5_contribution_pivot_summary": top5_pivot_summary,  
        "contribution_detailed_summary": pred_df_full,
        "top5_contribution_report": top5_report_df,
        "predictions_df": predictions_df,
        "primary_secondary_pollutants": primary_secondary_pollutants,  
        "fallback_candidates": fallback_df  
    }
    logging.info("=== Full post-processing analysis complete ===")
    return analysis_results

# ===================================================================
# Paper metrics utilities
# ===================================================================

def compute_paper_metrics(analysis_results: dict, dataset, config=None) -> dict:
    """Compute summary metrics for manuscript reporting.

    Metrics include:
    - Cosine similarity distribution between reconstructed and net pollution fingerprints
    - Proportion of receptors with cosine > 0.7 / > 0.8
    - Mean/std of contribution percentages by source type across receptors
    - Counts of receptors and sources (per type)
    - Top 10 receptors by Nemerow index
    """
    import numpy as np
    import pandas as pd
    from numpy.linalg import norm

    preds_full = analysis_results.get("contribution_detailed_summary")
    receptors_df = dataset.receptors_df.copy()
    sources_df = dataset.sources_df.copy()

    # Build unified background matrix for net pollution calculation
    backgrounds_df = dataset.unified_backgrounds_df.copy()
    metals = list(getattr(config, 'metals', ['As','Cd','Cr','Cu','Hg','Pb','Zn']))
    # Note: unified_backgrounds_df already has the correct column names (metals)
    # No need to rename columns as it's already in the correct format

    # Cosine similarity per receptor
    sims = []
    for rid in sorted(preds_full['receptor_idx'].dropna().astype(int).unique()):
        sub = preds_full[preds_full['receptor_idx'] == rid]
        if sub.empty:
            continue
        # weights per source (already normalized to percent per receptor)
        w = (sub['contribution_percent'].astype(float).values) / 100.0
        src_idx = sub['source_idx'].astype(int).values
        # source metal fingerprints
        fp = sources_df.loc[src_idx, metals].to_numpy(dtype=float)
        recon = (w.reshape(-1, 1) * fp).sum(axis=0)
        # target: net pollution fingerprint using unified background
        row = receptors_df.iloc[rid]
        bg_row = backgrounds_df.iloc[rid]
        tgt = []
        for m in metals:
            val = row.get(m, np.nan)
            bg = bg_row.get(m, np.nan)
            nv = (float(val) - float(bg)) if (pd.notna(val) and pd.notna(bg)) else np.nan
            tgt.append(max(0.0, nv) if pd.notna(nv) else 0.0)
        tgt = np.asarray(tgt, dtype=float)
        if norm(recon) == 0 or norm(tgt) == 0:
            sim = 0.0
        else:
            sim = float(np.dot(recon, tgt) / (norm(recon) * norm(tgt)))
        sims.append(sim)
    sims = np.asarray(sims, dtype=float)

    # Contribution summary by source type
    pivot = analysis_results.get('contribution_pivot_summary')
    contrib_means, contrib_stds = {}, {}
    if pivot is not None and not pivot.empty:
        # Keep common source types columns if present
        for col in ['atmosphere', 'irrigation', 'fertilizer', 'manure']:
            if col in pivot.columns:
                contrib_means[col] = float(pivot[col].mean())
                contrib_stds[col] = float(pivot[col].std())

    # Counts
    sources_by_type = sources_df['source_type'].value_counts().to_dict() if 'source_type' in sources_df.columns else {}

    # Top 10 receptors by Nemerow index
    receptors_with_nemerow = analysis_results.get('receptors_with_nemerow')
    top10 = []
    if receptors_with_nemerow is not None and not receptors_with_nemerow.empty and 'nemerow_index' in receptors_with_nemerow.columns:
        tmp = receptors_with_nemerow.nlargest(10, 'nemerow_index')
        for _, r in tmp[['receptor_sample', 'nemerow_index']].iterrows():
            top10.append({
                'receptor_sample': r['receptor_sample'],
                'nemerow_index': float(r['nemerow_index'])
            })

    metrics = {
        'cosine_mean': float(np.nanmean(sims)) if sims.size else None,
        'cosine_median': float(np.nanmedian(sims)) if sims.size else None,
        'cosine_q1': float(np.nanpercentile(sims, 25)) if sims.size else None,
        'cosine_q3': float(np.nanpercentile(sims, 75)) if sims.size else None,
        'n_receptors': int(len(set(preds_full['receptor_idx'].astype(int)))) if preds_full is not None else 0,
        'prop_gt_0_7': float((sims > 0.7).mean()) if sims.size else None,
        'prop_gt_0_8': float((sims > 0.8).mean()) if sims.size else None,
        'contrib_means': contrib_means,
        'contrib_stds': contrib_stds,
        'n_sources_total': int(len(sources_df)),
        'sources_by_type': {str(k): int(v) for k, v in sources_by_type.items()},
        'top10_by_nemerow': top10
    }
    return metrics
#  XAI Module: Gradient-based Explanation
# ===================================================================
def analyze_explanation(trainer, data, receptor_idx, top_k=10):
    """Analyzes model predictions for a specific receptor using gradient-based methods."""
    logging.info(f"\n=== Starting Gradient-based XAI analysis for receptor {receptor_idx} ===")
    try:
        # Ensure data is on the correct device
        data = data.to(trainer.device)
        trainer.model.eval()
        trainer.decoder.eval()
        data.x_receptor.requires_grad_(True)
        data.x_source.requires_grad_(True)
        if data.x_source.grad is not None: data.x_source.grad.zero_()

        # We need the encoder output, not the final model output
        h_receptor, _ = trainer.model(
            data.x_receptor, data.x_source,
            data.edge_index_rr, data.edge_index_rs,
            edge_attr_rr=getattr(data, 'edge_attr_rr', None),
            edge_attr_rs=getattr(data, 'edge_attr_rs', None)
        )

        target_embedding = h_receptor[receptor_idx]
        target_embedding.sum().backward(retain_graph=True)

        receptor_indices, source_indices = data.edge_index_rs[1], data.edge_index_rs[0]
        target_connections = receptor_indices == receptor_idx

        if not target_connections.any():
            logging.warning(f"Receptor {receptor_idx} has no source connections for XAI.")
            return None

        connected_sources = source_indices[target_connections]
        source_gradients = data.x_source.grad[connected_sources]

        importance_scores = torch.norm(source_gradients, p=2, dim=1).detach().cpu().numpy()

        explanation_data = []
        for i, (source_idx, importance) in enumerate(zip(connected_sources.cpu().numpy(), importance_scores)):
            if source_idx >= len(trainer.sources_df):
                logging.warning(f"Source index {source_idx} out of bounds in gradient explanation. Skipping.")
                continue

            source_info = trainer.sources_df.iloc[source_idx]
            # Apply terminology mapping for consistency
            source_type = source_info['source_type']
            if source_type == 'pesticide':
                source_type = 'fertilizer'
            explanation_data.append({'source_idx': source_idx, 'source_type': source_type, 'importance_score': importance, 'source_coords': (source_info['lon'], source_info['lat'])})

        explanation_data.sort(key=lambda x: x['importance_score'], reverse=True)

        logging.info(f"Top {min(top_k, len(explanation_data))} most important sources (Gradient-based):")
        for i, item in enumerate(explanation_data[:top_k]):
            logging.info(f"  {i+1}. Source {item['source_idx']} ({item['source_type']}) - Importance: {item['importance_score']:.4f}")

        return {'receptor_idx': receptor_idx, 'explanation_data': explanation_data}

    except Exception as e:
        logging.error(f"Gradient-based XAI analysis failed: {e}", exc_info=True)
        return None
    finally:
        if hasattr(data.x_receptor, 'requires_grad') and data.x_receptor.requires_grad:
            data.x_receptor.requires_grad_(False)
        if hasattr(data.x_source, 'requires_grad') and data.x_source.requires_grad:
            data.x_source.requires_grad_(False)

def run_gnnexplainer_analysis(trainer, data, receptor_idx, top_k=10, **kwargs):
    """
    Analyzes model predictions for a specific receptor using a simplified approach.
    Uses attention weights and gradient-based importance instead of GNNExplainer
    to avoid compatibility issues.
    """
    logging.info(f"\n=== Starting simplified explainer analysis for receptor {receptor_idx} ===")
    try:
        data = data.to(trainer.device)
        trainer.model.eval()

        # Get the receptor-source edges for the target receptor
        rs_edges = data.edge_index_rs
        target_receptor_mask = (rs_edges[0] == receptor_idx)

        if not target_receptor_mask.any():
            logging.warning(f"Receptor {receptor_idx} has no source connections.")
            return None

        connected_source_indices = rs_edges[1][target_receptor_mask]

        # Method 1: Use model attention weights if available
        with torch.no_grad():
            h_receptor, h_source = trainer.model(
                data.x_receptor, data.x_source,
                data.edge_index_rr, data.edge_index_rs
            )

            # Get attention weights from the model if available
            attention_weights = None
            if hasattr(trainer.model, 'get_attention_weights'):
                attention_weights = trainer.model.get_attention_weights()
            elif hasattr(trainer.model, 'attention_weights'):
                attention_weights = trainer.model.attention_weights

            # Method 2: Use gradient-based importance
            importance_scores = []

            # Enable gradients for input features
            data.x_receptor.requires_grad_(True)
            data.x_source.requires_grad_(True)

            # Forward pass to get receptor embedding
            h_receptor, h_source = trainer.model(
                data.x_receptor, data.x_source,
                data.edge_index_rr, data.edge_index_rs
            )

            # Get the target receptor's embedding
            target_embedding = h_receptor[receptor_idx]

            # Calculate importance based on embedding magnitude
            for source_idx in connected_source_indices:
                source_idx = source_idx.item()

                # Calculate distance-based importance
                source_embedding = h_source[source_idx]
                similarity = torch.cosine_similarity(
                    target_embedding.unsqueeze(0),
                    source_embedding.unsqueeze(0)
                ).item()

                # Use embedding norm as importance indicator
                source_norm = torch.norm(source_embedding).item()
                importance = similarity * source_norm

                importance_scores.append(importance)

            # Disable gradients
            data.x_receptor.requires_grad_(False)
            data.x_source.requires_grad_(False)

        # Prepare explanation data
        explanation_data = []
        for i, source_idx in enumerate(connected_source_indices):
            source_idx = source_idx.item()

            if source_idx >= len(trainer.sources_df):
                logging.warning(f"Source index {source_idx} out of bounds in attention explanation. Skipping.")
                continue

            source_info = trainer.sources_df.iloc[source_idx]

            # Apply terminology mapping for consistency
            source_type = source_info['source_type']
            if source_type == 'pesticide':
                source_type = 'fertilizer'

            explanation_data.append({
                'source_idx': source_idx,
                'source_type': source_type,
                'importance_score': importance_scores[i],
                'source_coords': (source_info['lon'], source_info['lat'])
            })

        # Sort by importance
        explanation_data.sort(key=lambda x: x['importance_score'], reverse=True)

        logging.info(f"Top {min(top_k, len(explanation_data))} most important sources:")
        for i, item in enumerate(explanation_data[:top_k]):
            logging.info(f"  {i+1}. Source {item['source_idx']} ({item['source_type']}) - Importance: {item['importance_score']:.4f}")

        return {
            'receptor_idx': receptor_idx,
            'explanation_data': explanation_data,
            'method': 'simplified_gradient_based'
        }

    except Exception as e:
        logging.error(f"Simplified explainer analysis failed: {e}", exc_info=True)
        return None