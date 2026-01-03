# Concurrent Performance Testing

This is a simplified concurrent testing tool for evaluating the concurrent performance of distributed systems.

## Design Philosophy

**Key Finding**: `~/trident/distributed-deploy/server.py` already supports concurrency natively!
- Uses `threading.Thread` to create independent threads for each client connection
- No need to modify server code
- Clients just need to send multiple concurrent queries

## File Description

- `config.py` - Configuration file (server IPs, concurrency levels, etc.)
- `concurrent_benchmark.py` - Main test script
- `README.md` - This file

## Usage

### 1. Start Servers (on 3 server machines)

```bash
# Server 1
cd ~/trident/distributed-deploy
python server.py --server-id 1 --dataset siftsmall

# Server 2
python server.py --server-id 2 --dataset siftsmall

# Server 3
python server.py --server-id 3 --dataset siftsmall
```

### 2. Run Concurrent Tests (on client machine)

```bash
cd ~/trident/concurrent-test

# Basic test (default concurrency levels: 1,2,4,8,16)
python concurrent_benchmark.py --dataset siftsmall --queries-per-level 50

# Custom concurrency levels
python concurrent_benchmark.py --dataset siftsmall \
  --concurrent-levels "1,2,4,8,16,32,64" \
  --queries-per-level 100

# Quick test
python concurrent_benchmark.py --dataset siftsmall \
  --concurrent-levels "1,2,4,8" \
  --queries-per-level 10
```

## Output Results

The test will output:
1. **Real-time Progress** - Query progress for each concurrency level
2. **Performance Metrics**:
   - Success Rate
   - Throughput (queries/sec)
   - Average Latency
   - P50/P95/P99 Latency
3. **Result Files** - `benchmark_results_*.json`

Example output:
```
====================================================================================================
Performance Summary
====================================================================================================
Concurrency  Success Rate  Throughput(qps)  Avg Latency(s)  P95 Latency(s)  P99 Latency(s)
----------------------------------------------------------------------------------------------------
1            100.0         0.85             1.176           1.200           1.250
2            100.0         1.65             1.212           1.280           1.320
4            100.0         3.12             1.282           1.450           1.520
8            100.0         5.89             1.358           1.680           1.780
16           100.0         10.24            1.562           2.100           2.350
====================================================================================================
```

## Testing Objectives

Verify concurrent characteristics of the system:
1. ✅ **Linear Scaling**: Throughput increases linearly with concurrency level
2. ✅ **Stable Latency**: Latency remains relatively stable (slight growth is normal)
3. ✅ **Saturation Point**: Find the concurrency level where system reaches maximum throughput
4. ✅ **Query Isolation**: No interference between different queries (verified by stable latency)

## Relationship with distributed-deploy

- **Reuse**: Directly uses the `DistributedClient` class from `distributed-deploy/client.py`
- **Simplification**: No need to modify server code
- **Focus**: Only implements concurrent testing logic
