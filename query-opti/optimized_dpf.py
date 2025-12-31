#!/usr/bin/env python3
"""
[CN]DPF[CN]，[CN]PRG[CN]calculate
"""
import hashlib
from typing import Tuple, Dict
import sys
sys.path.append('~/trident/standardDPF')

from standard_dpf import StandardDPF, DPFKey


class CachedPRG:
    """[CN]PRG[CN]"""
    
    def __init__(self, max_cache_size=10000):
        self.cache = {}
        self.max_cache_size = max_cache_size
        self.hits = 0
        self.misses = 0
        
    def expand(self, seed: bytes, seed_len: int) -> Tuple[bytes, bytes, int, int]:
        """[CN]PRG[CN]"""
        # [CN]seed[CN]
        cache_key = seed
        
        if cache_key in self.cache:
            self.hits += 1
            return self.cache[cache_key]
        
        self.misses += 1
        
        # calculatePRG[CN]
        # [CN]SHA-512[CN]SHA-256
        h = hashlib.sha512()
        h.update(seed)
        output = h.digest()
        
        # [CN]
        s_left = output[:seed_len]
        s_right = output[32:32+seed_len]
        t_left = output[seed_len] & 1
        t_right = output[32+seed_len] & 1
        
        # [CN]
        if len(self.cache) < self.max_cache_size:
            self.cache[cache_key] = (s_left, s_right, t_left, t_right)
        
        return s_left, s_right, t_left, t_right
    
    def get_stats(self):
        """[CN]"""
        total = self.hits + self.misses
        hit_rate = self.hits / total * 100 if total > 0 else 0
        return {
            'hits': self.hits,
            'misses': self.misses,
            'hit_rate': hit_rate,
            'cache_size': len(self.cache)
        }


class OptimizedDPF(StandardDPF):
    """[CN]DPF[CN]"""
    
    def __init__(self, domain_bits: int):
        super().__init__(domain_bits)
        self.prg_cache = CachedPRG()
        
    def _dpf_prg(self, seed: bytes) -> Tuple[bytes, bytes, int, int]:
        """[CN]PRG"""
        return self.prg_cache.expand(seed, self.seed_len)
    
    def eval_batch(self, key: DPFKey, positions: list) -> dict:
        """[CN]"""
        results = {}
        
        # [CN]，[CN]
        # [CN]，[CN]
        sorted_positions = sorted(positions)
        
        for x in sorted_positions:
            results[x] = self.eval(key, x)
        
        return results
    
    def get_prg_stats(self):
        """[CN]PRG[CN]"""
        return self.prg_cache.get_stats()


class OptimizedVDPF:
    """[CN]VDPF[CN]，[CN]DPF"""
    
    def __init__(self, domain_bits: int):
        self.domain_bits = domain_bits
        # [CN]DPF
        self.dpf = OptimizedDPF(domain_bits)
        
    def eval_batch_with_stats(self, key, positions: list) -> Tuple[dict, dict]:
        """[CN]return[CN]"""
        import time
        
        start_time = time.time()
        
        # [CN]
        self.dpf.prg_cache.hits = 0
        self.dpf.prg_cache.misses = 0
        
        # [CN]
        results = self.dpf.eval_batch(key, positions)
        
        eval_time = time.time() - start_time
        stats = self.dpf.get_prg_stats()
        stats['total_time'] = eval_time
        stats['positions_evaluated'] = len(positions)
        stats['avg_time_per_position'] = eval_time / len(positions) * 1000  # ms
        
        return results, stats