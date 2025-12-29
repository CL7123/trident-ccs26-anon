#!/usr/bin/env python3
"""
测试VDPF性能并进行详细分析
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
    """测试VDPF性能"""
    print(f"=== VDPF性能详细分析 ===")
    print(f"数据集: {dataset}")
    print(f"测试位置数: {num_positions}")
    
    # 1. 生成测试密钥
    print("\n1. 生成VDPF密钥...")
    wrapper = VDPFVectorWrapper(dataset_name=dataset)
    test_node_id = 1234
    keys = wrapper.generate_keys('node', test_node_id)
    
    # 反序列化密钥
    key1 = wrapper._deserialize_key(keys[0])
    
    # 2. 使用带性能分析的包装器
    print("\n2. 创建性能分析器...")
    profiled_wrapper = ProfiledBatchWrapper(dataset)
    
    # 3. 测试单个评估的性能
    print("\n3. 测试单个VDPF评估...")
    single_start = time.time()
    result = profiled_wrapper.vdpf.eval(1, key1, 100)
    single_time = time.time() - single_start
    print(f"单个评估耗时: {single_time*1000:.3f}ms")
    
    # 4. 测试批量评估的性能（带详细分析）
    print("\n4. 测试批量VDPF评估（带详细分析）...")
    results = profiled_wrapper.eval_batch_with_profiling(key1, 0, num_positions, 1)
    
    # 5. 使用cProfile进行更深入的分析
    print("\n5. 使用cProfile进行深入分析...")
    profiler = cProfile.Profile()
    
    # 分析1000个位置的评估
    profiler.enable()
    for i in range(1000):
        profiled_wrapper.vdpf.eval(1, key1, i)
    profiler.disable()
    
    # 打印最耗时的函数
    print("\n最耗时的20个函数:")
    stats = pstats.Stats(profiler)
    stats.sort_stats('cumulative')
    stats.print_stats(20)
    
    # 6. 测试不同批次大小的性能
    print("\n6. 测试不同批次大小的性能...")
    batch_sizes = [1, 10, 100, 500, 1000]
    
    for batch_size in batch_sizes:
        profiled_wrapper.vdpf.reset_timers()
        
        batch_start = time.time()
        for i in range(batch_size):
            profiled_wrapper.vdpf.eval(1, key1, i)
        batch_time = time.time() - batch_start
        
        print(f"\n批次大小 {batch_size}:")
        print(f"  总时间: {batch_time*1000:.1f}ms")
        print(f"  平均每个: {batch_time/batch_size*1000:.3f}ms")
        print(f"  吞吐量: {batch_size/batch_time:.0f} ops/sec")
    
    # 7. 分析内存访问模式
    print("\n7. 分析内存访问模式...")
    # 连续访问 vs 随机访问
    positions_sequential = list(range(1000))
    positions_random = np.random.randint(0, 10000, 1000).tolist()
    
    # 测试连续访问
    seq_start = time.time()
    for pos in positions_sequential:
        profiled_wrapper.vdpf.eval(1, key1, pos)
    seq_time = time.time() - seq_start
    
    # 测试随机访问
    rand_start = time.time()
    for pos in positions_random:
        profiled_wrapper.vdpf.eval(1, key1, pos)
    rand_time = time.time() - rand_start
    
    print(f"连续访问1000个位置: {seq_time*1000:.1f}ms")
    print(f"随机访问1000个位置: {rand_time*1000:.1f}ms")
    print(f"差异: {(rand_time/seq_time - 1)*100:.1f}%")


if __name__ == "__main__":
    # 运行性能测试
    test_vdpf_performance(dataset="siftsmall", num_positions=1000)