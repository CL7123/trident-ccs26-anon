"""
VDPF评估性能分析器
用于详细测量VDPF评估过程中各个步骤的耗时
"""
import time
import sys
sys.path.append('~/trident/standardDPF')
sys.path.append('~/trident/query-opti')

from vdpf_23 import VDPF23, VDPF23Key
from vdpf_batch_optimizer import OptimizedBatchVDPFWrapper
import numpy as np


class ProfiledVDPF23(VDPF23):
    """带性能分析的VDPF23"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.reset_timers()
        
    def reset_timers(self):
        """重置所有计时器"""
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
        """带性能测量的eval"""
        start_time = time.time()
        self.timers['eval_calls'] += 1
        
        # 原始逻辑，但添加时间测量
        g = fb.g_key
        k = fb.k_key
        
        if b == 1:
            b_g, b_k = 0, 0
        elif b == 2:
            b_g, b_k = 1, 0
        else:  # b == 3
            b_g, b_k = 1, 1
        
        # 测量VDPF+ G评估
        t1 = time.time()
        y_g = self.vdpf_plus_g.eval(b_g, g, x)
        self.timers['vdpf_plus_g_time'] += time.time() - t1
        
        # 测量VDPF+ K评估
        t2 = time.time()
        y_k = self.vdpf_plus_k.eval(b_k, k, x)
        self.timers['vdpf_plus_k_time'] += time.time() - t2
        
        # 测量XOR操作
        t3 = time.time()
        y_xor = y_g ^ y_k
        self.timers['xor_operations'] += time.time() - t3
        
        # 测量embed_mult操作
        t4 = time.time()
        y = self.op_embed_mult(b, y_xor)
        self.timers['embed_mult'] += time.time() - t4
        
        self.timers['eval_total_time'] += time.time() - start_time
        return y
        
    def get_timer_report(self):
        """获取计时报告"""
        if self.timers['eval_calls'] == 0:
            return "No evaluations performed"
            
        avg_per_call = self.timers['eval_total_time'] / self.timers['eval_calls'] * 1000  # ms
        
        report = f"""
VDPF评估性能分析报告:
- 总调用次数: {self.timers['eval_calls']}
- 总耗时: {self.timers['eval_total_time']*1000:.1f}ms
- 平均每次调用: {avg_per_call:.3f}ms

时间分解:
- VDPF+ G评估: {self.timers['vdpf_plus_g_time']*1000:.1f}ms ({self.timers['vdpf_plus_g_time']/self.timers['eval_total_time']*100:.1f}%)
- VDPF+ K评估: {self.timers['vdpf_plus_k_time']*1000:.1f}ms ({self.timers['vdpf_plus_k_time']/self.timers['eval_total_time']*100:.1f}%)
- XOR操作: {self.timers['xor_operations']*1000:.1f}ms ({self.timers['xor_operations']/self.timers['eval_total_time']*100:.1f}%)
- Embed乘法: {self.timers['embed_mult']*1000:.1f}ms ({self.timers['embed_mult']/self.timers['eval_total_time']*100:.1f}%)
"""
        return report


class ProfiledBatchWrapper:
    """带性能分析的批量VDPF包装器"""
    
    def __init__(self, dataset_name):
        # 使用ProfiledVDPF23而不是普通的VDPF23
        self.vdpf = ProfiledVDPF23(
            domain_bits=17 if dataset_name == "siftsmall" else 20,
            security_param=128,
            dataset_name=dataset_name
        )
        self.batch_wrapper = OptimizedBatchVDPFWrapper(self.vdpf)
        
    def eval_batch_with_profiling(self, key, start_pos: int, end_pos: int, party_id: int):
        """带性能分析的批量评估"""
        batch_start_time = time.time()
        
        # 重置计时器
        self.vdpf.reset_timers()
        
        # 执行批量评估
        results = {}
        batch_size = end_pos - start_pos
        
        # 测量不同大小的子批次
        sub_batch_sizes = [1, 10, 100, min(1000, batch_size)]
        
        for sub_batch_size in sub_batch_sizes:
            if sub_batch_size > batch_size:
                continue
                
            sub_batch_start = time.time()
            
            # 评估一个子批次
            for i in range(start_pos, min(start_pos + sub_batch_size, end_pos)):
                results[i] = self.vdpf.eval(party_id, key, i)
                
            sub_batch_time = time.time() - sub_batch_start
            print(f"\n子批次大小 {sub_batch_size}: {sub_batch_time*1000:.1f}ms, "
                  f"平均每个: {sub_batch_time/sub_batch_size*1000:.3f}ms")
        
        # 评估剩余的
        for i in range(start_pos + sub_batch_sizes[-1], end_pos):
            results[i] = self.vdpf.eval(party_id, key, i)
        
        batch_total_time = time.time() - batch_start_time
        
        # 打印性能报告
        print(self.vdpf.get_timer_report())
        print(f"\n批量评估总时间: {batch_total_time*1000:.1f}ms")
        print(f"批量大小: {batch_size}")
        print(f"平均每个位置: {batch_total_time/batch_size*1000:.3f}ms")
        
        return results