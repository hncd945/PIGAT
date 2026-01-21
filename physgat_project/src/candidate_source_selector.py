#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Candidate source selector module for multi-stage filtering of pollution sources.
Implements distance-based, chemical similarity, and physical constraint filtering for edge construction.

候选污染源筛选模块，实现多阶段污染源过滤。
基于距离、化学相似性和物理约束进行边构建的筛选逻辑。

Author: Wenhao Wang
"""

import logging
import numpy as np
import pandas as pd
import torch
from sklearn.metrics.pairwise import cosine_similarity
from typing import Dict, List, Tuple, Optional


class CandidateSourceSelector:
    """
    Selects candidate pollution sources for each receptor using multi-stage filtering.
    
    The selection process consists of 5 stages:
    1. Distance-based pre-filtering
    2. Multi-dimensional scoring (distance, chemical, strength, wind, hydro)
    3. Top-K selection per source type
    4. Global diversity optimization (min/max candidates)
    5. Edge weight calculation
    
    This class encapsulates the complex logic from the original 400-line function
    into smaller, testable, and maintainable methods.
    """
    
    def __init__(self, config, receptors_df, sources_df, receptor_backgrounds, metals):
        """
        Initialize the candidate source selector.
        
        Args:
            config: Configuration object (OmegaConf DictConfig)
            receptors_df: DataFrame of receptor locations and concentrations
            sources_df: DataFrame of source locations, types, and concentrations
            receptor_backgrounds: Tensor of background concentrations for receptors
            metals: List of metal element names
        """
        self.config = config
        self.receptors_df = receptors_df
        self.sources_df = sources_df
        self.receptor_backgrounds = receptor_backgrounds
        self.metals = metals
        
        # Precompute receptor and source profiles
        self.receptor_profiles_contrib_only = torch.relu(
            torch.tensor(receptors_df[metals].values) - receptor_backgrounds
        ).numpy()
        self.source_chem_profiles_np = sources_df[metals].values
        
        # Load configuration
        self._load_config()
        
        logging.info("CandidateSourceSelector initialized")
        logging.info(f"  Receptors: {len(receptors_df)}")
        logging.info(f"  Sources: {len(sources_df)}")
        logging.info(f"  Metals: {metals}")
        logging.info(f"  Distance thresholds: {self.max_dist_config}")
        logging.info(f"  Global min/max candidates: {self.global_min_candidates}/{self.global_max_candidates}")
    
    def _load_config(self):
        """Load and validate configuration parameters."""
        # Get candidate selection configuration
        candidate_config = self.config.graph.get('candidate_selection', {})
        self.max_dist_config = self.config.graph.get('max_distance_km', {})
        self.default_max_dist = self.max_dist_config.get('default', 5)
        
        # Scoring weights configuration
        self.scoring_weights_config = candidate_config.get('scoring_weights', {})
        self.zero_candidate_config = candidate_config.get('zero_candidate_exception', {})
        
        # Candidate selection parameters
        self.max_candidates_per_type = candidate_config.get('max_candidates_per_type', 8)
        self.min_candidates_per_type = candidate_config.get('min_candidates_per_type', 4)
        self.global_min_candidates = candidate_config.get('global_min_candidates', 16)
        self.global_max_candidates = candidate_config.get('global_max_candidates', 12)
        self.max_total_connections = candidate_config.get('max_total_connections_per_receptor', 32)
        self.min_total_connections = candidate_config.get('min_total_connections_per_receptor', 16)
        self.use_strict_filtering = candidate_config.get('use_strict_filtering', True)
        
        # Fix configuration conflicts
        if self.global_max_candidates < self.global_min_candidates:
            logging.warning(
                f"Configuration conflict: global_max_candidates ({self.global_max_candidates}) < "
                f"global_min_candidates ({self.global_min_candidates})")
            logging.warning(f"Adjusting global_min_candidates to {self.global_max_candidates}")
            self.global_min_candidates = self.global_max_candidates
        
        if self.min_total_connections > self.global_max_candidates:
            logging.warning(
                f"Configuration conflict: min_total_connections ({self.min_total_connections}) > "
                f"global_max_candidates ({self.global_max_candidates})")
            logging.warning(f"Adjusting min_total_connections to {self.global_max_candidates}")
            self.min_total_connections = self.global_max_candidates
        
        # Numerical parameters
        self.numerical_eps = getattr(self.config.graph, 'numerical_epsilon', 1e-8)
        self.min_wind_weight = getattr(self.config.graph, 'min_wind_weight', 0.1)
    
    def get_scoring_weights(self, source_type: str, is_exception: bool = False) -> Dict[str, float]:
        """
        Get scoring weights for a specific source type.

        Args:
            source_type: Type of source ('atmosphere', 'irrigation', 'fertilizer', 'manure')
            is_exception: Whether this is an exception case (no candidates in distance filtering)
        
        Returns:
            Dictionary of scoring weights for different score components
        """
        if is_exception and self.zero_candidate_config.get('enabled', True):
            # Use exception weights when no candidates found in distance filtering
            exception_key = f"{source_type}_exception"
            if exception_key in self.zero_candidate_config:
                return self.zero_candidate_config[exception_key]
            else:
                # Fallback: reduce distance weight and redistribute
                normal_weights = self.scoring_weights_config.get(
                    source_type, 
                    self.scoring_weights_config.get('default', {})
                )
                distance_reduction = self.zero_candidate_config.get('distance_weight_reduction', 0.3)
                exception_weights = normal_weights.copy()
                
                if 'distance_score' in exception_weights:
                    old_distance = exception_weights['distance_score']
                    exception_weights['distance_score'] = distance_reduction
                    # Redistribute the reduced distance weight to other factors
                    redistribution = (old_distance - distance_reduction) / max(1, len(exception_weights) - 1)
                    for key in exception_weights:
                        if key != 'distance_score':
                            exception_weights[key] += redistribution
                
                return exception_weights
        else:
            # Use normal weights
            return self.scoring_weights_config.get(source_type, self.scoring_weights_config.get('default', {
                'distance_score': 0.5,
                'chemical_score': 0.2,
                'strength_score': 0.15,
                'wind_score': 0.15
            }))
    
    def select_candidates_for_receptor(
        self,
        receptor_idx: int,
        receptor_coord: np.ndarray,
        source_coords: np.ndarray,
        wind_influence: Optional[np.ndarray] = None,
        distance_calculator = None
    ) -> Dict[str, Dict]:
        """
        Select candidate sources for a single receptor.
        
        This is the main entry point that orchestrates the 5-stage selection process.
        
        Args:
            receptor_idx: Index of the receptor
            receptor_coord: Coordinates of the receptor [lon, lat]
            source_coords: Array of source coordinates
            wind_influence: Optional wind influence matrix for atmospheric sources
            distance_calculator: Function to calculate distance (for DEM-based distance)
        
        Returns:
            Dictionary mapping source types to their selected candidates and scores
        """
        source_type_edges = {}
        
        # Iterate through each source type
        for source_type in self.sources_df['source_type'].unique():
            source_indices_of_type = self.sources_df[
                self.sources_df['source_type'] == source_type
            ].index
            
            if len(source_indices_of_type) == 0:
                continue
            
            # Stage 1: Distance-based pre-filtering
            distance_filtered_indices, distance_values, is_exception = self._filter_by_distance(
                receptor_idx, receptor_coord, source_type, source_indices_of_type,
                source_coords, distance_calculator
            )
            
            if not distance_filtered_indices:
                continue
            
            # Stage 2: Multi-dimensional scoring
            candidates_df = self._score_candidates(
                receptor_idx, source_type, distance_filtered_indices, distance_values,
                wind_influence, is_exception
            )
            
            # Stage 3: Top-K selection per source type
            selected_candidates = self._select_top_k_per_type(
                receptor_idx, source_type, candidates_df, source_coords,
                receptor_coord, distance_calculator
            )
            
            if len(selected_candidates) == 0:
                continue
            
            # Store candidate information
            source_type_edges[source_type] = {
                'indices': selected_candidates.index.tolist(),
                'scores': selected_candidates,
                'effective_k': len(selected_candidates),
                'total_filtered': len(distance_filtered_indices),
                'distance_filtered': len(distance_filtered_indices),
                'final_selected': len(selected_candidates)
            }
        
        # Stage 4: Global diversity optimization
        if source_type_edges:
            source_type_edges = self._optimize_diversity(
                receptor_idx, receptor_coord, source_type_edges, source_coords,
                distance_calculator
            )
        
        return source_type_edges
    
    def _filter_by_distance(
        self,
        receptor_idx: int,
        receptor_coord: np.ndarray,
        source_type: str,
        source_indices: pd.Index,
        source_coords: np.ndarray,
        distance_calculator = None
    ) -> Tuple[List[int], List[float], bool]:
        """
        Stage 1: Filter sources by distance threshold.

        Args:
            receptor_idx: Index of the receptor
            receptor_coord: Coordinates of the receptor
            source_type: Type of source
            source_indices: Indices of sources of this type
            source_coords: Array of all source coordinates
            distance_calculator: Function to calculate distance

        Returns:
            Tuple of (filtered_indices, distance_values, is_exception_case)
        """
        # Get distance threshold for this source type
        max_dist = self.max_dist_config.get(source_type, self.default_max_dist)

        distance_filtered_indices = []
        distance_values = []

        # Apply strict distance filtering
        for s_idx in source_indices:
            source_coord = source_coords[s_idx]

            # Calculate distance based on source type
            if distance_calculator is not None:
                # Use provided distance calculator (for DEM-based distance)
                dist = distance_calculator(receptor_coord, source_coord, source_type)
            else:
                # Fallback to Euclidean distance
                try:
                    from .data_utils import _calculate_distance
                except ImportError:
                    from data_utils import _calculate_distance
                dist = _calculate_distance(
                    receptor_coord[1], receptor_coord[0],
                    source_coord[1], source_coord[0]
                )

            # Only include sources within distance threshold
            if dist <= max_dist:
                distance_filtered_indices.append(s_idx)
                distance_values.append(dist)

        # Handle zero candidate exception
        is_exception_case = False
        if not distance_filtered_indices:
            # STRICT RULE: manure sources NEVER allow exception mechanism
            if source_type == 'manure':
                logging.info(
                    f"Receptor {receptor_idx} {source_type} sources: No sources within {max_dist}km. "
                    f"STRICT POLICY: No exception allowed for manure sources, skipping."
                )
                return [], [], False

            # For other source types, apply exception mechanism if enabled
            if self.zero_candidate_config.get('enabled', True):
                # Apply exception mechanism (expand search)
                exception_result = self._apply_exception_mechanism(
                    receptor_idx, receptor_coord, source_type, source_indices,
                    source_coords, max_dist, distance_calculator
                )

                if exception_result:
                    distance_filtered_indices, distance_values, is_exception_case = exception_result
                    logging.info(
                        f"Receptor {receptor_idx} {source_type} sources: "
                        f"Applied exception mechanism, found {len(distance_filtered_indices)} candidates"
                    )
                else:
                    logging.debug(
                        f"Receptor {receptor_idx} {source_type} sources: "
                        f"No sources found through exception mechanism"
                    )
                    return [], [], False
            else:
                logging.debug(
                    f"Receptor {receptor_idx} {source_type} sources: "
                    f"No sources within {max_dist}km, skipping"
                )
                return [], [], False

        return distance_filtered_indices, distance_values, is_exception_case

    def _apply_exception_mechanism(
        self,
        receptor_idx: int,
        receptor_coord: np.ndarray,
        source_type: str,
        source_indices: pd.Index,
        source_coords: np.ndarray,
        original_max_dist: float,
        distance_calculator = None
    ) -> Optional[Tuple[List[int], List[float], bool]]:
        """
        Apply exception mechanism when no candidates found within distance threshold.

        Expands the search radius to find at least 1 candidate for non-manure sources.

        Args:
            receptor_idx: Index of the receptor
            receptor_coord: Coordinates of the receptor
            source_type: Type of source
            source_indices: Indices of sources of this type
            source_coords: Array of source coordinates
            original_max_dist: Original distance threshold
            distance_calculator: Function to calculate distance

        Returns:
            Tuple of (filtered_indices, distance_values, is_exception=True) or None
        """
        # Define maximum expansion limits per source type
        max_expansion_limits = {
            'atmosphere': 10.0,  # Can expand up to 10km
            'irrigation': 5.0,   # Can expand up to 5km
            'fertilizer': 3.0,   # Can expand up to 3km
            'manure': 0.2       # Never expand (strict 0.2km limit)
        }

        max_allowed_dist = max_expansion_limits.get(source_type, 5.0)

        all_distances = []
        for s_idx in source_indices:
            source_coord = source_coords[s_idx]

            # Calculate distance
            if distance_calculator is not None:
                dist = distance_calculator(receptor_coord, source_coord, source_type)
            else:
                try:
                    from .data_utils import _calculate_distance
                except ImportError:
                    from data_utils import _calculate_distance
                dist = _calculate_distance(
                    receptor_coord[1], receptor_coord[0],
                    source_coord[1], source_coord[0]
                )

            # Only include sources within the maximum allowed expansion distance
            if dist <= max_allowed_dist:
                all_distances.append((dist, s_idx))

        if not all_distances:
            logging.warning(
                f"Receptor {receptor_idx} {source_type}: No sources found within "
                f"maximum expansion distance {max_allowed_dist}km"
            )
            return None

        # Sort by distance and take the closest one (minimum 1 candidate)
        all_distances.sort(key=lambda x: x[0])
        min_required = 1  # Only find 1 candidate in exception case
        expanded_indices = [idx for _, idx in all_distances[:min_required]]
        expanded_distances = [dist for dist, _ in all_distances[:min_required]]

        logging.info(
            f"Expanded search for {source_type}: found {len(expanded_indices)} candidates "
            f"within extended range (max: {max_allowed_dist}km)"
        )

        return expanded_indices, expanded_distances, True  # is_exception=True

    def _score_candidates(
        self,
        receptor_idx: int,
        source_type: str,
        candidate_indices: List[int],
        distance_values: List[float],
        wind_influence: Optional[np.ndarray],
        is_exception: bool
    ) -> pd.DataFrame:
        """
        Stage 2: Calculate multi-dimensional scores for candidates.

        Scores include:
        - Distance score (closer = higher)
        - Chemical similarity score (cosine similarity)
        - Source strength score (total concentration)
        - Wind direction score (atmosphere only)
        - Hydrological score (irrigation only)

        Args:
            receptor_idx: Index of the receptor
            source_type: Type of source
            candidate_indices: Indices of candidate sources
            distance_values: Distance values for each candidate
            wind_influence: Wind influence matrix
            is_exception: Whether this is an exception case

        Returns:
            DataFrame with scores for each candidate
        """
        candidates_df = pd.DataFrame(index=candidate_indices)

        # 1. Distance score (closer = higher score)
        max_dist_in_candidates = max(distance_values) if distance_values else 1.0
        # Handle case where all distances are 0 (sources at same location as receptor)
        if max_dist_in_candidates == 0:
            candidates_df['distance_score'] = [1.0] * len(distance_values)
        else:
            candidates_df['distance_score'] = [
                (max_dist_in_candidates - d) / max_dist_in_candidates
                for d in distance_values
            ]

        # 2. Chemical similarity score (cosine similarity)
        chem_scores = cosine_similarity(
            [self.receptor_profiles_contrib_only[receptor_idx]],
            self.source_chem_profiles_np[candidate_indices]
        )[0]
        candidates_df['chemical_score'] = chem_scores

        # 3. Source strength score (total concentration)
        source_concentration_sum = self.sources_df.loc[candidate_indices, self.metals].sum(axis=1)
        max_conc = source_concentration_sum.max()
        candidates_df['strength_score'] = source_concentration_sum / (max_conc + self.numerical_eps)

        # 4. Wind direction score (atmosphere only)
        if source_type == 'atmosphere' and wind_influence is not None:
            wind_scores = []
            for idx in candidate_indices:
                wind_score = wind_influence[idx, receptor_idx]
                wind_scores.append(max(wind_score, self.min_wind_weight))
            candidates_df['wind_score'] = wind_scores
        else:
            # For non-atmosphere sources, set default wind score
            candidates_df['wind_score'] = 1.0

        # 5. Hydrological score (irrigation only)
        if source_type == 'irrigation':
            try:
                hydro_scores = self._calculate_hydrological_influence(
                    candidate_indices, receptor_idx
                )
                candidates_df['hydro_score'] = hydro_scores
            except Exception as e:
                logging.warning(
                    f"Hydrological influence calculation failed, fallback to distance score: {e}"
                )
                candidates_df['hydro_score'] = candidates_df['distance_score']
        else:
            candidates_df['hydro_score'] = candidates_df['distance_score']

        # Calculate weighted total score
        scoring_weights = self.get_scoring_weights(source_type, is_exception)
        total_score = np.zeros(len(candidates_df))

        # Apply weights based on source type and available scores
        if 'distance_score' in scoring_weights:
            total_score += candidates_df['distance_score'] * scoring_weights['distance_score']
        if 'chemical_score' in scoring_weights:
            total_score += candidates_df['chemical_score'] * scoring_weights['chemical_score']
        if 'strength_score' in scoring_weights:
            total_score += candidates_df['strength_score'] * scoring_weights['strength_score']
        if 'wind_score' in scoring_weights and source_type == 'atmosphere':
            total_score += candidates_df['wind_score'] * scoring_weights['wind_score']
        if 'hydro_score' in scoring_weights and source_type == 'irrigation':
            total_score += candidates_df['hydro_score'] * scoring_weights['hydro_score']

        candidates_df['total_score'] = total_score

        # Log scoring weights used
        if is_exception:
            logging.info(
                f"Receptor {receptor_idx} {source_type} sources: "
                f"Using exception weights: {scoring_weights}"
            )
        else:
            logging.info(
                f"Receptor {receptor_idx} {source_type} sources: "
                f"Using normal weights: {scoring_weights}"
            )

        # Sort by total score
        sorted_candidates = candidates_df.sort_values('total_score', ascending=False)

        return sorted_candidates

    def _calculate_hydrological_influence(
        self,
        source_indices: List[int],
        receptor_idx: int
    ) -> List[float]:
        """
        Calculate hydrological influence scores for irrigation sources.

        Considers:
        - Flow direction alignment
        - Distance to stream
        - Flow accumulation

        Args:
            source_indices: Indices of source candidates
            receptor_idx: Index of the receptor

        Returns:
            List of hydrological scores
        """
        hydro_scores = []
        receptor_coord = self.receptors_df.iloc[receptor_idx][['lon', 'lat']].values

        for source_idx in source_indices:
            source_coord = self.sources_df.iloc[source_idx][['lon', 'lat']].values

            # Get hydrological features
            source_flow_dir = self.sources_df.iloc[source_idx].get('flow_direction', 0)
            source_flow_acc = self.sources_df.iloc[source_idx].get('flow_accumulation', 0)
            source_stream_dist = self.sources_df.iloc[source_idx].get('distance_to_stream', 1000)

            # Calculate flow direction alignment
            direction_vector = receptor_coord - source_coord
            direction_angle = np.arctan2(direction_vector[1], direction_vector[0]) * 180 / np.pi
            flow_alignment = np.cos(np.radians(direction_angle - source_flow_dir))
            flow_score = max(flow_alignment, 0.1)

            # Calculate stream proximity score
            stream_score = np.exp(-source_stream_dist / 1000)

            # Calculate flow accumulation score
            flow_acc_score = min(source_flow_acc / 1000, 1.0)

            # Weighted combination
            hydro_score = 0.4 * flow_score + 0.3 * stream_score + 0.3 * flow_acc_score
            hydro_scores.append(max(hydro_score, 0.1))

        return hydro_scores

    def _select_top_k_per_type(
        self,
        receptor_idx: int,
        source_type: str,
        candidates_df: pd.DataFrame,
        source_coords: np.ndarray,
        receptor_coord: np.ndarray,
        distance_calculator = None
    ) -> pd.DataFrame:
        """
        Stage 3: Select top-K candidates per source type.

        Args:
            receptor_idx: Index of the receptor
            source_type: Type of source
            candidates_df: DataFrame with scored candidates
            source_coords: Array of source coordinates
            receptor_coord: Coordinates of the receptor
            distance_calculator: Function to calculate distance

        Returns:
            DataFrame with selected top-K candidates
        """
        available_candidates = len(candidates_df)

        # Handle zero candidates case
        if available_candidates == 0:
            # STRICT RULE: manure sources NEVER allow out-of-range search
            if source_type == 'manure':
                logging.info(
                    f"Receptor {receptor_idx} {source_type}: No candidates in range. "
                    f"STRICT POLICY: No out-of-range search allowed for manure sources."
                )
                return pd.DataFrame()
            else:
                # For other source types: already handled in _filter_by_distance
                # This should not happen, but return empty DataFrame as fallback
                logging.warning(
                    f"Receptor {receptor_idx} {source_type}: No candidates available "
                    f"(should have been handled in distance filtering)"
                )
                return pd.DataFrame()

        # Handle fewer candidates than minimum
        elif available_candidates < self.min_candidates_per_type:
            # NEW POLICY: Do NOT expand search just because we have fewer than minimum
            # Only expand when we have ZERO candidates (handled above)
            # If we have some candidates within range, use them even if fewer than minimum
            if source_type == 'manure':
                logging.info(
                    f"Receptor {receptor_idx} {source_type}: Only {available_candidates} candidates in range. "
                    f"STRICT POLICY: No expansion allowed for manure sources."
                )
            else:
                logging.info(
                    f"Receptor {receptor_idx} {source_type}: Only {available_candidates} candidates in range "
                    f"(min: {self.min_candidates_per_type}). NEW POLICY: Using available candidates within range, "
                    f"no expansion."
                )

        # Select candidates: at most max_candidates_per_type
        if self.use_strict_filtering:
            target_count = min(self.max_candidates_per_type, available_candidates)
            selected_candidates = candidates_df.head(target_count)
        else:
            selected_candidates = candidates_df

        # Log selection
        logging.info(
            f"Receptor {receptor_idx} {source_type} sources: {len(selected_candidates)} candidates selected "
            f"(max: {self.max_candidates_per_type}, available: {available_candidates})"
        )

        return selected_candidates

    def _optimize_diversity(
        self,
        receptor_idx: int,
        receptor_coord: np.ndarray,
        source_type_edges: Dict[str, Dict],
        source_coords: np.ndarray,
        distance_calculator = None
    ) -> Dict[str, Dict]:
        """
        Stage 4: Optimize source type diversity.

        Ensures:
        - At least global_min_candidates total candidates
        - At most global_max_candidates total candidates
        - Respects distance constraints

        Args:
            receptor_idx: Index of the receptor
            receptor_coord: Coordinates of the receptor
            source_type_edges: Dictionary of selected candidates per source type
            source_coords: Array of source coordinates
            distance_calculator: Function to calculate distance

        Returns:
            Optimized source_type_edges dictionary
        """
        # Calculate total connections for this receptor
        total_connections = sum(
            edge_data['final_selected']
            for edge_data in source_type_edges.values()
        )

        # Check global minimum candidates
        if total_connections < self.global_min_candidates:
            logging.info(
                f"Receptor {receptor_idx}: Only {total_connections} candidates, "
                f"need {self.global_min_candidates}. Adding more candidates..."
            )
            source_type_edges = self._ensure_global_min_candidates(
                receptor_idx, receptor_coord, source_type_edges, source_coords,
                distance_calculator
            )

            # Recompute total connections
            total_connections = sum(
                edge_data['final_selected']
                for edge_data in source_type_edges.values()
            )

        # Check global maximum candidates
        if total_connections > self.global_max_candidates:
            logging.info(
                f"Receptor {receptor_idx}: Too many candidates ({total_connections}), "
                f"reducing to {self.global_max_candidates}..."
            )
            source_type_edges = self._reduce_to_global_max_candidates(
                receptor_idx, receptor_coord, source_type_edges, source_coords,
                distance_calculator
            )

            # Recompute total connections
            total_connections = sum(
                edge_data['final_selected']
                for edge_data in source_type_edges.values()
            )

        # Apply maximum connection constraint
        if self.use_strict_filtering and total_connections > self.max_total_connections:
            logging.info(
                f"Receptor {receptor_idx}: Applying strict filtering, "
                f"reducing from {total_connections} to {self.max_total_connections}..."
            )
            source_type_edges = self._apply_max_total_connections(
                receptor_idx, receptor_coord, source_type_edges, source_coords,
                distance_calculator
            )

        logging.debug(f"Receptor {receptor_idx} final connections: {total_connections}")

        # Record if no candidates pass filtering
        if total_connections == 0:
            logging.info(
                f"Receptor {receptor_idx}: No candidates passed filtering, "
                f"no connections will be established"
            )

        return source_type_edges

    def _ensure_global_min_candidates(
        self,
        receptor_idx: int,
        receptor_coord: np.ndarray,
        source_type_edges: Dict[str, Dict],
        source_coords: np.ndarray,
        distance_calculator = None
    ) -> Dict[str, Dict]:
        """
        Ensure at least global_min_candidates total candidates.

        Adds additional candidates from unselected sources if needed.

        Args:
            receptor_idx: Index of the receptor
            receptor_coord: Coordinates of the receptor
            source_type_edges: Current selected candidates
            source_coords: Array of source coordinates
            distance_calculator: Function to calculate distance

        Returns:
            Updated source_type_edges with additional candidates
        """
        current_total = sum(
            edge_data['final_selected']
            for edge_data in source_type_edges.values()
        )
        needed_candidates = self.global_min_candidates - current_total

        if needed_candidates <= 0:
            return source_type_edges

        # Get already selected indices
        selected_indices = set()
        for edge_data in source_type_edges.values():
            selected_indices.update(edge_data['indices'])

        # Find unselected sources from all types
        all_unselected = []
        for source_type in self.sources_df['source_type'].unique():
            source_indices_of_type = self.sources_df[
                self.sources_df['source_type'] == source_type
            ].index

            # Get max distance for this source type
            max_dist_for_type = self.max_dist_config.get(source_type, self.default_max_dist)

            for s_idx in source_indices_of_type:
                if s_idx not in selected_indices:
                    source_coord = source_coords[s_idx]

                    # Calculate distance
                    if distance_calculator is not None:
                        dist = distance_calculator(receptor_coord, source_coord, source_type)
                    else:
                        try:
                            from .data_utils import _calculate_distance
                        except ImportError:
                            from data_utils import _calculate_distance
                        dist = _calculate_distance(
                            receptor_coord[1], receptor_coord[0],
                            source_coord[1], source_coord[0]
                        )

                    # STRICT RULE: manure sources must be within 0.2km, no exceptions
                    if source_type == 'manure' and dist > 0.2:
                        continue

                    # For other source types, also respect their distance limits
                    if dist > max_dist_for_type:
                        continue

                    # Simple score: closer is better
                    score = 1.0 / (1.0 + dist) + np.random.random() * 0.1
                    all_unselected.append((score, source_type, s_idx, dist))

        # Sort by score and take top needed_candidates
        all_unselected.sort(key=lambda x: x[0], reverse=True)
        additional_candidates = all_unselected[:needed_candidates]

        # Add additional candidates to source_type_edges
        for score, source_type, s_idx, dist in additional_candidates:
            if source_type not in source_type_edges:
                source_type_edges[source_type] = {
                    'indices': [],
                    'scores': pd.DataFrame(),
                    'effective_k': 0,
                    'total_filtered': 0,
                    'distance_filtered': 0,
                    'final_selected': 0
                }

            # Add to indices
            source_type_edges[source_type]['indices'].append(s_idx)
            source_type_edges[source_type]['final_selected'] += 1

            # Add simple scores to scores DataFrame
            scores_df = source_type_edges[source_type]['scores']
            if scores_df.empty:
                scores_df = pd.DataFrame(index=[s_idx])

            # Compute simple scores
            distance_score = 1.0 / (1.0 + dist)
            strength_series = self.sources_df[self.metals].sum(axis=1)
            max_conc = strength_series.max() if len(strength_series) > 0 else 1.0
            strength_score = float(strength_series.loc[s_idx] / (max_conc + self.numerical_eps))
            wind_score = 1.0 if source_type == 'atmosphere' else 0.0
            hydro_score = distance_score

            scores_df.loc[s_idx, ['distance_score', 'chemical_score', 'strength_score',
                                  'wind_score', 'hydro_score']] = [
                distance_score, 0.5, strength_score, wind_score, hydro_score
            ]

            # Calculate total score
            scoring_weights = self.get_scoring_weights(source_type, False)
            total_score = (
                distance_score * scoring_weights.get('distance_score', 0.5) +
                0.5 * scoring_weights.get('chemical_score', 0.2) +
                strength_score * scoring_weights.get('strength_score', 0.15)
            )
            scores_df.loc[s_idx, 'total_score'] = total_score

            source_type_edges[source_type]['scores'] = scores_df

            logging.debug(
                f"Receptor {receptor_idx}: Added {source_type} source {s_idx} "
                f"at {dist:.2f}km (score: {score:.3f})"
            )

        logging.info(
            f"Receptor {receptor_idx}: Added {len(additional_candidates)} candidates "
            f"to meet global minimum"
        )

        return source_type_edges

    def _reduce_to_global_max_candidates(
        self,
        receptor_idx: int,
        receptor_coord: np.ndarray,
        source_type_edges: Dict[str, Dict],
        source_coords: np.ndarray,
        distance_calculator = None
    ) -> Dict[str, Dict]:
        """
        Reduce total candidates to global_max_candidates.

        Keeps candidates with highest total scores across all source types.

        Args:
            receptor_idx: Index of the receptor
            receptor_coord: Coordinates of the receptor
            source_type_edges: Current selected candidates
            source_coords: Array of source coordinates
            distance_calculator: Function to calculate distance

        Returns:
            Updated source_type_edges with reduced candidates
        """
        # Collect all candidates with their scores
        all_candidates = []
        for source_type, edge_data in source_type_edges.items():
            for idx in edge_data['indices']:
                total_score = edge_data['scores'].loc[idx, 'total_score']
                all_candidates.append({
                    'source_type': source_type,
                    'source_idx': idx,
                    'total_score': total_score
                })

        # Sort by total score (descending)
        all_candidates.sort(key=lambda x: x['total_score'], reverse=True)

        # Keep only top global_max_candidates
        selected_candidates = all_candidates[:self.global_max_candidates]

        # Rebuild source_type_edges with selected candidates
        new_source_type_edges = {}
        for source_type in source_type_edges.keys():
            new_source_type_edges[source_type] = {
                'indices': [],
                'scores': pd.DataFrame(),
                'effective_k': 0,
                'total_filtered': source_type_edges[source_type]['total_filtered'],
                'distance_filtered': source_type_edges[source_type]['distance_filtered'],
                'final_selected': 0
            }

        for candidate in selected_candidates:
            source_type = candidate['source_type']
            idx = candidate['source_idx']

            new_source_type_edges[source_type]['indices'].append(idx)
            new_source_type_edges[source_type]['final_selected'] += 1

            # Copy scores
            if new_source_type_edges[source_type]['scores'].empty:
                new_source_type_edges[source_type]['scores'] = source_type_edges[source_type]['scores'].loc[[idx]]
            else:
                new_source_type_edges[source_type]['scores'] = pd.concat([
                    new_source_type_edges[source_type]['scores'],
                    source_type_edges[source_type]['scores'].loc[[idx]]
                ])

        # Remove empty source types
        new_source_type_edges = {
            k: v for k, v in new_source_type_edges.items()
            if v['final_selected'] > 0
        }

        logging.info(
            f"Receptor {receptor_idx}: Reduced from {len(all_candidates)} to "
            f"{len(selected_candidates)} candidates"
        )

        return new_source_type_edges

    def _apply_max_total_connections(
        self,
        receptor_idx: int,
        receptor_coord: np.ndarray,
        source_type_edges: Dict[str, Dict],
        source_coords: np.ndarray,
        distance_calculator = None
    ) -> Dict[str, Dict]:
        """
        Apply maximum total connections constraint.

        Similar to _reduce_to_global_max_candidates but uses max_total_connections.

        Args:
            receptor_idx: Index of the receptor
            receptor_coord: Coordinates of the receptor
            source_type_edges: Current selected candidates
            source_coords: Array of source coordinates
            distance_calculator: Function to calculate distance

        Returns:
            Updated source_type_edges with reduced candidates
        """
        # Collect all candidates with distance verification
        all_candidates = []
        for source_type, edge_data in source_type_edges.items():
            # Get max distance for this source type
            max_dist_for_type = self.max_dist_config.get(source_type, self.default_max_dist)

            for idx in edge_data['indices']:
                # Calculate distance to verify it's within limits
                source_coord = source_coords[idx]

                if distance_calculator is not None:
                    dist = distance_calculator(receptor_coord, source_coord, source_type)
                else:
                    try:
                        from .data_utils import _calculate_distance
                    except ImportError:
                        from data_utils import _calculate_distance
                    dist = _calculate_distance(
                        receptor_coord[1], receptor_coord[0],
                        source_coord[1], source_coord[0]
                    )

                # STRICT RULE: manure sources must be within 0.2km, no exceptions
                if source_type == 'manure' and dist > 0.2:
                    logging.warning(
                        f"Receptor {receptor_idx}: Skipping manure source {idx} "
                        f"at {dist:.2f}km (exceeds 0.2km limit)"
                    )
                    continue

                # For other source types, also respect their distance limits
                if dist > max_dist_for_type:
                    logging.warning(
                        f"Receptor {receptor_idx}: Skipping {source_type} source {idx} "
                        f"at {dist:.2f}km (exceeds {max_dist_for_type}km limit)"
                    )
                    continue

                score = edge_data['scores'].loc[idx, 'total_score']
                all_candidates.append((score, source_type, idx))

        # Sort by score and take top max_total_connections
        all_candidates.sort(key=lambda x: x[0], reverse=True)
        selected_candidates = all_candidates[:self.max_total_connections]

        # Rebuild source_type_edges with selected candidates
        new_source_type_edges = {}
        for score, source_type, idx in selected_candidates:
            if source_type not in new_source_type_edges:
                new_source_type_edges[source_type] = {
                    'indices': [],
                    'scores': source_type_edges[source_type]['scores'],
                    'final_selected': 0
                }
            new_source_type_edges[source_type]['indices'].append(idx)
            new_source_type_edges[source_type]['final_selected'] += 1

        logging.debug(
            f"Receptor {receptor_idx}: Reduced connections from {len(all_candidates)} to "
            f"{len(selected_candidates)} (max: {self.max_total_connections})"
        )

        return new_source_type_edges

    def calculate_edge_weights(
        self,
        source_type_edges: Dict[str, Dict],
        receptor_idx: int,
        receptor_coords: np.ndarray,
        source_coords: np.ndarray
    ) -> Tuple[List[Tuple[int, int]], List[float], List[float]]:
        """
        Stage 5: Calculate edge weights for selected candidates.

        Args:
            source_type_edges: Dictionary of selected candidates per source type
            receptor_idx: Index of the receptor
            receptor_coords: Array of receptor coordinates
            source_coords: Array of source coordinates

        Returns:
            Tuple of (edges, edge_weights, wind_features)
        """
        all_edges = []
        edge_weights = []
        wind_features = []

        for source_type, edge_data in source_type_edges.items():
            # Get source-type specific weights for edge weight calculation
            edge_scoring_weights = self.get_scoring_weights(source_type, False)

            for source_idx in edge_data['indices']:
                all_edges.append((receptor_idx, source_idx))

                scores = edge_data['scores']

                # Get individual scores
                distance_score = scores.loc[source_idx, 'distance_score']
                chemical_score = scores.loc[source_idx, 'chemical_score']
                strength_score = scores.loc[source_idx, 'strength_score']
                wind_score = scores.loc[source_idx, 'wind_score'] if 'wind_score' in scores.columns else 0.0
                hydro_score = scores.loc[source_idx, 'hydro_score'] if 'hydro_score' in scores.columns else distance_score

                # Calculate weighted final weight
                final_weight = 0.0
                if 'distance_score' in edge_scoring_weights:
                    final_weight += distance_score * edge_scoring_weights['distance_score']
                if 'chemical_score' in edge_scoring_weights:
                    final_weight += chemical_score * edge_scoring_weights['chemical_score']
                if 'strength_score' in edge_scoring_weights:
                    final_weight += strength_score * edge_scoring_weights['strength_score']
                if 'wind_score' in edge_scoring_weights and source_type == 'atmosphere':
                    final_weight += wind_score * edge_scoring_weights['wind_score']
                if 'hydro_score' in edge_scoring_weights and source_type == 'irrigation':
                    final_weight += hydro_score * edge_scoring_weights['hydro_score']

                edge_weights.append(max(final_weight, 0.01))

                # Calculate wind direction alignment feature for atmospheric sources
                if source_type == 'atmosphere':
                    receptor_coord = receptor_coords[receptor_idx]
                    source_coord = source_coords[source_idx]

                    # Get prevailing wind direction from config
                    env_cfg = self.config.get('environment', {}) if hasattr(self.config, 'get') else {}
                    prevailing_wind = env_cfg.get('prevailing_wind_direction', 225)

                    # Calculate wind direction feature
                    try:
                        from .data_utils import _calculate_wind_direction_feature
                    except ImportError:
                        from data_utils import _calculate_wind_direction_feature
                    wind_alignment = _calculate_wind_direction_feature(
                        receptor_coord, source_coord, prevailing_wind
                    )
                    wind_features.append(wind_alignment)
                else:
                    wind_features.append(0.0)

        return all_edges, edge_weights, wind_features

