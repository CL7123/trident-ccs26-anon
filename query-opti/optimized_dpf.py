#!/usr/bin/env python3
"""
Optimized DPF implementation using PRG cache to reduce redundant computations
"""
import hashlib
from typing import Tuple, Dict
import sys
sys.path.append('~/trident/standardDPF')

from standard_dpf import StandardDPF, DPFKey


class CachedPRG:
    """PRG implementation with caching"""

    def __init__(self, max_cache_size=10000):
        self.cache = {}
        self.max_cache_size = max_cache_size
        self.hits = 0
        self.misses = 0

    def expand(self, seed: bytes, seed_len: int) -> Tuple[bytes, bytes, int, int]:
        """PRG expansion with caching"""
        # Use seed as cache key
        cache_key = seed

        if cache_key in self.cache:
            self.hits += 1
            return self.cache[cache_key]

        self.misses += 1

        # Calculate PRG expansion
        # Use single SHA-512 instead of two SHA-256 calls
        h = hashlib.sha512()
        h.update(seed)
        output = h.digest()

        # Split output
        s_left = output[:seed_len]
        s_right = output[32:32+seed_len]
        t_left = output[seed_len] & 1
        t_right = output[32+seed_len] & 1

        # Cache result
        if len(self.cache) < self.max_cache_size:
            self.cache[cache_key] = (s_left, s_right, t_left, t_right)

        return s_left, s_right, t_left, t_right

    def get_stats(self):
        """Get cache statistics"""
        total = self.hits + self.misses
        hit_rate = self.hits / total * 100 if total > 0 else 0
        return {
            'hits': self.hits,
            'misses': self.misses,
            'hit_rate': hit_rate,
            'cache_size': len(self.cache)
        }


class OptimizedDPF(StandardDPF):
    """Optimized DPF implementation"""

    def __init__(self, domain_bits: int):
        super().__init__(domain_bits)
        self.prg_cache = CachedPRG()

    def _dpf_prg(self, seed: bytes) -> Tuple[bytes, bytes, int, int]:
        """PRG with caching"""
        return self.prg_cache.expand(seed, self.seed_len)

    def eval_batch(self, key: DPFKey, positions: list) -> dict:
        """Batch evaluate multiple positions"""
        results = {}

        # Group by common prefix to improve cache hit rate
        # First sort positions, adjacent positions are more likely to share path prefixes
        sorted_positions = sorted(positions)

        for x in sorted_positions:
            results[x] = self.eval(key, x)

        return results

    def get_prg_stats(self):
        """Get PRG cache statistics"""
        return self.prg_cache.get_stats()


class OptimizedVDPF:
    """Optimized VDPF implementation using optimized DPF"""

    def __init__(self, domain_bits: int):
        self.domain_bits = domain_bits
        # Use optimized DPF
        self.dpf = OptimizedDPF(domain_bits)

    def eval_batch_with_stats(self, key, positions: list) -> Tuple[dict, dict]:
        """Batch evaluate and return statistical information"""
        import time

        start_time = time.time()

        # Reset statistics
        self.dpf.prg_cache.hits = 0
        self.dpf.prg_cache.misses = 0

        # Batch evaluate
        results = self.dpf.eval_batch(key, positions)

        eval_time = time.time() - start_time
        stats = self.dpf.get_prg_stats()
        stats['total_time'] = eval_time
        stats['positions_evaluated'] = len(positions)
        stats['avg_time_per_position'] = eval_time / len(positions) * 1000  # ms

        return results, stats