"""
VDPF batch evaluation optimizer
Reduce function call overhead and improve cache efficiency through batch processing
"""
import numpy as np
from typing import List, Dict
import sys
sys.path.append('~/trident/standardDPF')

from vdpf_23 import VDPF23, VDPF23Key


class BatchVDPF23:
    """Batch optimized version of VDPF23"""

    def __init__(self, vdpf23_instance: VDPF23):
        self.vdpf = vdpf23_instance

    def eval_batch(self, b: int, fb: VDPF23Key, positions: List[int]) -> Dict[int, int]:
        """
        Batch evaluate VDPF values at multiple positions

        Args:
            b: Party ID (1, 2, or 3)
            fb: The key for party b
            positions: List of positions to evaluate

        Returns:
            Dictionary mapping position to value
        """
        # Pre-parse key (done only once)
        g = fb.g_key
        k = fb.k_key

        # Pre-calculate party-related constants (done only once)
        if b == 1:
            b_g, b_k = 0, 0
        elif b == 2:
            b_g, b_k = 1, 0
        else:  # b == 3
            b_g, b_k = 1, 1

        results = {}

        # Batch processing, improve cache locality
        for x in positions:
            # Evaluate VDPF+
            y_g = self.vdpf.vdpf_plus_g.eval(b_g, g, x)
            y_k = self.vdpf.vdpf_plus_k.eval(b_k, k, x)

            # Calculate final result
            y_xor = y_g ^ y_k
            y = self.vdpf.op_embed_mult(b, y_xor)

            results[x] = y

        return results

    def eval_batch_vectorized(self, b: int, fb: VDPF23Key, positions: np.ndarray) -> np.ndarray:
        """
        Vectorized batch evaluation (if supported by underlying implementation)

        Args:
            b: Party ID (1, 2, or 3)
            fb: The key for party b
            positions: Numpy array of positions

        Returns:
            Numpy array of values
        """
        # Pre-parse key
        g = fb.g_key
        k = fb.k_key

        # Pre-calculate party-related constants
        if b == 1:
            b_g, b_k = 0, 0
        elif b == 2:
            b_g, b_k = 1, 0
        else:  # b == 3
            b_g, b_k = 1, 1

        # Pre-allocate result array
        results = np.zeros(len(positions), dtype=np.uint64)

        # Batch evaluation
        for i, x in enumerate(positions):
            y_g = self.vdpf.vdpf_plus_g.eval(b_g, g, x)
            y_k = self.vdpf.vdpf_plus_k.eval(b_k, k, x)
            y_xor = y_g ^ y_k
            results[i] = self.vdpf.op_embed_mult(b, y_xor)

        return results


class OptimizedBatchVDPFWrapper:
    """Optimized batch VDPF wrapper"""

    def __init__(self, vdpf_instance):
        self.vdpf = vdpf_instance
        self.batch_evaluator = BatchVDPF23(vdpf_instance)

    def eval_positions_batch(self, key, positions: List[int], party_id: int) -> Dict[int, int]:
        """
        Batch evaluate multiple positions

        Args:
            key: VDPF key
            positions: List of positions
            party_id: Party ID

        Returns:
            Mapping from position to value
        """
        return self.batch_evaluator.eval_batch(party_id, key, positions)

    def eval_range_batch(self, key, start: int, end: int, party_id: int) -> Dict[int, int]:
        """
        Batch evaluate all positions in a range

        Args:
            key: VDPF key
            start: Start position (inclusive)
            end: End position (exclusive)
            party_id: Party ID

        Returns:
            Mapping from position to value
        """
        positions = list(range(start, end))
        return self.batch_evaluator.eval_batch(party_id, key, positions)
