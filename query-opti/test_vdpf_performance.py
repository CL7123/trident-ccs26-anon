#!/usr/bin/env python3
"""
[CN]VDPF[CN]
"""
import sys
import time
import numpy as np
sys.path.append('~/trident/query-opti')
sys.path.append('~/trident/src')

from vdpf_profiler import ProfiledBatchWrapper
from dpf_wrapper import VDPFVectorWrapper
from binary_serializer import BinaryKeySerializer
import cProfile
import pstats


def test_vdpf_performance(dataset="siftsmall", num_positions=1000):
    """[CN]VDPF[CN]"""
    print(f"=== VDPF[CN] ===")
    print(f"Dataset: {dataset}")
    print(f"[CN]: {num_positions}")
    
    # 1. [CN]
    print("\n1. [CN]VDPF[CN]...")
    wrapper = VDPFVectorWrapper(dataset_name=dataset)
    test_node_id = 1234
    keys = wrapper.generate_keys('node', test_node_id)
    
    # Deserialize keys
    key1 = wrapper._deserialize_key(keys[0])
    
    # 2. [CN]
    print("\n2. create[CN]...")
    profiled_wrapper = ProfiledBatchWrapper(dataset)
    
    # 3. [CN]
    print("\n3. [CN]VDPF[CN]...")
    single_start = time.time()
    result = profiled_wrapper.vdpf.eval(1, key1, 100)
    single_time = time.time() - single_start
    print(f"[CN]: {single_time*1000:.3f}ms")
    
    # 4. [CN]（[CN]）
    print("\n4. [CN]VDPF[CN]（[CN]）...")
    results = profiled_wrapper.eval_batch_with_profiling(key1, 0, num_positions, 1)
    
    # 5. [CN]cProfile[CN]
    print("\n5. [CN]cProfile[CN]...")
    profiler = cProfile.Profile()
    
    # [CN]1000[CN]
    profiler.enable()
    for i in range(1000):
        profiled_wrapper.vdpf.eval(1, key1, i)
    profiler.disable()
    
    # print[CN]
    print("\n[CN]20[CN]:")
    stats = pstats.Stats(profiler)
    stats.sort_stats('cumulative')
    stats.print_stats(20)
    
    # 6. [CN]
    print("\n6. [CN]...")
    batch_sizes = [1, 10, 100, 500, 1000]
    
    for batch_size in batch_sizes:
        profiled_wrapper.vdpf.reset_timers()
        
        batch_start = time.time()
        for i in range(batch_size):
            profiled_wrapper.vdpf.eval(1, key1, i)
        batch_time = time.time() - batch_start
        
        print(f"\n[CN] {batch_size}:")
        print(f"  [CN]: {batch_time*1000:.1f}ms")
        print(f"  [CN]: {batch_time/batch_size*1000:.3f}ms")
        print(f"  [CN]: {batch_size/batch_time:.0f} ops/sec")
    
    # 7. [CN]
    print("\n7. [CN]...")
    # [CN] vs [CN]
    positions_sequential = list(range(1000))
    positions_random = np.random.randint(0, 10000, 1000).tolist()
    
    # [CN]
    seq_start = time.time()
    for pos in positions_sequential:
        profiled_wrapper.vdpf.eval(1, key1, pos)
    seq_time = time.time() - seq_start
    
    # [CN]
    rand_start = time.time()
    for pos in positions_random:
        profiled_wrapper.vdpf.eval(1, key1, pos)
    rand_time = time.time() - rand_start
    
    print(f"[CN]1000[CN]: {seq_time*1000:.1f}ms")
    print(f"[CN]1000[CN]: {rand_time*1000:.1f}ms")
    print(f"[CN]: {(rand_time/seq_time - 1)*100:.1f}%")


if __name__ == "__main__":
    # [CN]
    test_vdpf_performance(dataset="siftsmall", num_positions=1000)