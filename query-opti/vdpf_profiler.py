"""
VDPF evaluation performance profiler
Used for detailed measurement of time spent in each step of VDPF evaluation process
"""
import time
import sys
sys.path.append('~/trident/standardDPF')
sys.path.append('~/trident/query-opti')

from vdpf_23 import VDPF23, VDPF23Key
from vdpf_batch_optimizer import OptimizedBatchVDPFWrapper
import numpy as np


class ProfiledVDPF23(VDPF23):
    """VDPF23 with performance profiling"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.reset_timers()

    def reset_timers(self):
        """Reset all timers"""
        self.timers = {
            'eval_calls': 0,
            'eval_total_time': 0.0,
            'vdpf_plus_g_time': 0.0,
            'vdpf_plus_k_time': 0.0,
            'xor_operations': 0.0,
            'embed_mult': 0.0,
            'prg_time': 0.0,
            'tree_traversal': 0.0
        }

    def eval(self, b: int, fb: VDPF23Key, x: int) -> int:
        """Eval with performance measurement"""
        start_time = time.time()
        self.timers['eval_calls'] += 1

        # Original logic, but with time measurement
        g = fb.g_key
        k = fb.k_key

        if b == 1:
            b_g, b_k = 0, 0
        elif b == 2:
            b_g, b_k = 1, 0
        else:  # b == 3
            b_g, b_k = 1, 1

        # Measure VDPF+ G evaluation
        t1 = time.time()
        y_g = self.vdpf_plus_g.eval(b_g, g, x)
        self.timers['vdpf_plus_g_time'] += time.time() - t1

        # Measure VDPF+ K evaluation
        t2 = time.time()
        y_k = self.vdpf_plus_k.eval(b_k, k, x)
        self.timers['vdpf_plus_k_time'] += time.time() - t2

        # Measure XOR operation
        t3 = time.time()
        y_xor = y_g ^ y_k
        self.timers['xor_operations'] += time.time() - t3

        # Measure embed_mult operation
        t4 = time.time()
        y = self.op_embed_mult(b, y_xor)
        self.timers['embed_mult'] += time.time() - t4

        self.timers['eval_total_time'] += time.time() - start_time
        return y

    def get_timer_report(self):
        """Get timing report"""
        if self.timers['eval_calls'] == 0:
            return "No evaluations performed"

        avg_per_call = self.timers['eval_total_time'] / self.timers['eval_calls'] * 1000  # ms

        report = f"""
VDPF Evaluation Performance Profiling Report:
- Total calls: {self.timers['eval_calls']}
- Total time: {self.timers['eval_total_time']*1000:.1f}ms
- Average per call: {avg_per_call:.3f}ms

Time breakdown:
- VDPF+ G evaluation: {self.timers['vdpf_plus_g_time']*1000:.1f}ms ({self.timers['vdpf_plus_g_time']/self.timers['eval_total_time']*100:.1f}%)
- VDPF+ K evaluation: {self.timers['vdpf_plus_k_time']*1000:.1f}ms ({self.timers['vdpf_plus_k_time']/self.timers['eval_total_time']*100:.1f}%)
- XOR operations: {self.timers['xor_operations']*1000:.1f}ms ({self.timers['xor_operations']/self.timers['eval_total_time']*100:.1f}%)
- Embed multiplication: {self.timers['embed_mult']*1000:.1f}ms ({self.timers['embed_mult']/self.timers['eval_total_time']*100:.1f}%)
"""
        return report


class ProfiledBatchWrapper:
    """Batch VDPF wrapper with performance profiling"""

    def __init__(self, dataset_name):
        # Use ProfiledVDPF23 instead of normal VDPF23
        self.vdpf = ProfiledVDPF23(
            domain_bits=17 if dataset_name == "siftsmall" else 20,
            security_param=128,
            dataset_name=dataset_name
        )
        self.batch_wrapper = OptimizedBatchVDPFWrapper(self.vdpf)

    def eval_batch_with_profiling(self, key, start_pos: int, end_pos: int, party_id: int):
        """Batch evaluation with performance profiling"""
        batch_start_time = time.time()

        # Reset timers
        self.vdpf.reset_timers()

        # Execute batch evaluation
        results = {}
        batch_size = end_pos - start_pos

        # Measure different size sub-batches
        sub_batch_sizes = [1, 10, 100, min(1000, batch_size)]

        for sub_batch_size in sub_batch_sizes:
            if sub_batch_size > batch_size:
                continue

            sub_batch_start = time.time()

            # Evaluate a sub-batch
            for i in range(start_pos, min(start_pos + sub_batch_size, end_pos)):
                results[i] = self.vdpf.eval(party_id, key, i)

            sub_batch_time = time.time() - sub_batch_start
            print(f"\nSub-batch size {sub_batch_size}: {sub_batch_time*1000:.1f}ms, "
                  f"avg per: {sub_batch_time/sub_batch_size*1000:.3f}ms")

        # Evaluate remaining
        for i in range(start_pos + sub_batch_sizes[-1], end_pos):
            results[i] = self.vdpf.eval(party_id, key, i)

        batch_total_time = time.time() - batch_start_time

        # Print performance report
        print(self.vdpf.get_timer_report())
        print(f"\nBatch evaluation total time: {batch_total_time*1000:.1f}ms")
        print(f"Batch size: {batch_size}")
        print(f"Average per position: {batch_total_time/batch_size*1000:.3f}ms")

        return results
