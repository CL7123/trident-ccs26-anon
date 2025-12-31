# concurrentperformance test

[CN]yes[CN]simplification[CN]concurrenttest[CN],[CN]testdistributedsystem[CN]concurrentperformance.

## [CN]

**[CN]**: `~/trident/distributed-deploy/server.py` [CN]supportconcurrent!
- usage `threading.Thread` [CN]clientconnectcreateindependentthread
- [CN]modifyservercode
- [CN]clientconcurrentsend[CN]query[CN]

## filedescription

- `config.py` - profile(serverIP, concurrent[CN])
- `concurrent_benchmark.py` - [CN]test[CN]
- `README.md` - [CN]file

## usagemethod

### 1. startserver([CN]3[CN]serveron)

```bash
# server1
cd ~/trident/distributed-deploy
python server.py --server-id 1 --dataset siftsmall

# server2
python server.py --server-id 2 --dataset siftsmall

# server3
python server.py --server-id 3 --dataset siftsmall
```

### 2. runconcurrenttest([CN]client[CN]on)

```bash
cd ~/trident/concurrent-test

# basictest(defaultconcurrent[CN]: 1,2,4,8,16)
python concurrent_benchmark.py --dataset siftsmall --queries-per-level 50

# customconcurrent[CN]
python concurrent_benchmark.py --dataset siftsmall \
  --concurrent-levels "1,2,4,8,16,32,64" \
  --queries-per-level 100

# fasttest
python concurrent_benchmark.py --dataset siftsmall \
  --concurrent-levels "1,2,4,8" \
  --queries-per-level 10
```

## outputresult

test[CN]output:
1. **real-time[CN]degree** - [CN]concurrent[CN]query[CN]degree
2. **performancemetrics**:
   - success[CN]
   - throughput (queries/sec)
   - averagelate
   - P50/P95/P99late
3. **resultfile** - `benchmark_results_*.json`

exampleoutput:
```
====================================================================================================
performance[CN]
====================================================================================================
concurrent[CN]     success[CN]      throughput(qps)      averagelate(s)      P95late(s)      P99late(s)
----------------------------------------------------------------------------------------------------
1            100.0       0.85            1.176           1.200           1.250
2            100.0       1.65            1.212           1.280           1.320
4            100.0       3.12            1.282           1.450           1.520
8            100.0       5.89            1.358           1.680           1.780
16           100.0       10.24           1.562           2.100           2.350
====================================================================================================
```

## testobjective

verificationsystem[CN]concurrentattribute:
1. ✅ **[CN]extension**: throughput[CN]concurrent[CN]
2. ✅ **stablelate**: late[CN]stable([CN]yespositive[CN])
3. ✅ **[CN]dot**: [CN]system[CN]maximumthroughput[CN]concurrent[CN]
4. ✅ **queryisolation**: [CN]query[CN]([CN]stable[CN]lateverification)

## [CN] distributed-deploy [CN]relation

- **[CN]**: [CN]usage `distributed-deploy/client.py` in[CN] `DistributedClient` class
- **simplification**: [CN]modifyservercode
- **[CN]**: [CN]implementationconcurrenttest[CN]
