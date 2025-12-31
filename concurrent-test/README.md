# [CN]test

[CN]test[CN]，[CN]test[CN]。

## [CN]

**[CN]**: `~/trident/distributed-deploy/server.py` [CN]！
- [CN] `threading.Thread` [CN]client[CN]
- [CN]server[CN]
- [CN]client[CN]query[CN]

## file[CN]

- `config.py` - configurefile（serverIP、[CN]）
- `concurrent_benchmark.py` - [CN]test[CN]
- `README.md` - [CN]file

## [CN]

### 1. startserver（[CN]3[CN]server[CN]）

```bash
# server1
cd ~/trident/distributed-deploy
python server.py --server-id 1 --dataset siftsmall

# server2
python server.py --server-id 2 --dataset siftsmall

# server3
python server.py --server-id 3 --dataset siftsmall
```

### 2. run[CN]test（[CN]client[CN]）

```bash
cd ~/trident/concurrent-test

# [CN]test（[CN]: 1,2,4,8,16）
python concurrent_benchmark.py --dataset siftsmall --queries-per-level 50

# [CN]
python concurrent_benchmark.py --dataset siftsmall \
  --concurrent-levels "1,2,4,8,16,32,64" \
  --queries-per-level 100

# [CN]test
python concurrent_benchmark.py --dataset siftsmall \
  --concurrent-levels "1,2,4,8" \
  --queries-per-level 10
```

## outputresult

test[CN]output：
1. **[CN]** - [CN]query[CN]
2. **[CN]**:
   - [CN]
   - [CN] (queries/sec)
   - [CN]
   - P50/P95/P99[CN]
3. **resultfile** - `benchmark_results_*.json`

[CN]output：
```
====================================================================================================
[CN]
====================================================================================================
[CN]     [CN]      [CN](qps)      [CN](s)      P95[CN](s)      P99[CN](s)
----------------------------------------------------------------------------------------------------
1            100.0       0.85            1.176           1.200           1.250
2            100.0       1.65            1.212           1.280           1.320
4            100.0       3.12            1.282           1.450           1.520
8            100.0       5.89            1.358           1.680           1.780
16           100.0       10.24           1.562           2.100           2.350
====================================================================================================
```

## test[CN]

[CN]：
1. ✅ **[CN]**: [CN]
2. ✅ **[CN]**: [CN]（[CN]）
3. ✅ **[CN]**: [CN]
4. ✅ **query[CN]**: [CN]query[CN]（[CN]）

## [CN] distributed-deploy [CN]

- **[CN]**: [CN] `distributed-deploy/client.py` [CN] `DistributedClient` [CN]
- **[CN]**: [CN]server[CN]
- **[CN]**: [CN]test[CN]
