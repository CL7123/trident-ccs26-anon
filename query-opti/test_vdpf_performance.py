#!/usr/bin/env python3
"""
Test VDPF performance and conduct detailed analysis
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
    """Test VDPF performance"""
    print(f"=== VDPF performance detailed analysis ===")
    print(f"Dataset: {dataset}")
    print(f"Test positions: {num_positions}")

    # 1. Generate test key
    print("\n1. Generating VDPF key...")
    wrapper = VDPFVectorWrapper(dataset_name=dataset)
    test_node_id = 1234
    keys = wrapper.generate_keys('node', test_node_id)

    # Deserialize key
    key1 = wrapper._deserialize_key(keys[0])

    # 2. Use performance analyzer wrapper
    print("\n2. Creating performance analyzer...")
    profiled_wrapper = ProfiledBatchWrapper(dataset)

    # 3. Test single evaluation performance
    print("\n3. Testing single VDPF evaluation...")
    single_start = time.time()
    result = profiled_wrapper.vdpf.eval(1, key1, 100)
    single_time = time.time() - single_start
    print(f"Single evaluation time: {single_time*1000:.3f}ms")

    # 4. Test batch evaluation performance (with detailed analysis)
    print("\n4. Testing batch VDPF evaluation (with detailed analysis)...")
    results = profiled_wrapper.eval_batch_with_profiling(key1, 0, num_positions, 1)

    # 5. Use cProfile for deeper analysis
    print("\n5. Using cProfile for in-depth analysis...")
    profiler = cProfile.Profile()

    # Analyze evaluation of 1000 positions
    profiler.enable()
    for i in range(1000):
        profiled_wrapper.vdpf.eval(1, key1, i)
    profiler.disable()

    # Print most time-consuming functions
    print("\nTop 20 most time-consuming functions:")
    stats = pstats.Stats(profiler)
    stats.sort_stats('cumulative')
    stats.print_stats(20)

    # 6. Test different batch sizes performance
    print("\n6. Testing different batch sizes performance...")
    batch_sizes = [1, 10, 100, 500, 1000]

    for batch_size in batch_sizes:
        profiled_wrapper.vdpf.reset_timers()

        batch_start = time.time()
        for i in range(batch_size):
            profiled_wrapper.vdpf.eval(1, key1, i)
        batch_time = time.time() - batch_start

        print(f"\nBatch size {batch_size}:")
        print(f"  Total time: {batch_time*1000:.1f}ms")
        print(f"  Average per: {batch_time/batch_size*1000:.3f}ms")
        print(f"  Throughput: {batch_size/batch_time:.0f} ops/sec")

    # 7. Analyze memory access pattern
    print("\n7. Analyzing memory access pattern...")
    # Sequential access vs random access
    positions_sequential = list(range(1000))
    positions_random = np.random.randint(0, 10000, 1000).tolist()

    # Test sequential access
    seq_start = time.time()
    for pos in positions_sequential:
        profiled_wrapper.vdpf.eval(1, key1, pos)
    seq_time = time.time() - seq_start

    # Test random access
    rand_start = time.time()
    for pos in positions_random:
        profiled_wrapper.vdpf.eval(1, key1, pos)
    rand_time = time.time() - rand_start

    print(f"Sequential access 1000 positions: {seq_time*1000:.1f}ms")
    print(f"Random access 1000 positions: {rand_time*1000:.1f}ms")
    print(f"Difference: {(rand_time/seq_time - 1)*100:.1f}%")


if __name__ == "__main__":
    # Run performance test
    test_vdpf_performance(dataset="siftsmall", num_positions=1000)
