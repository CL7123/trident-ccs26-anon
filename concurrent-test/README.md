# 并发性能测试

这是一个简化的并发测试工具，用于测试分布式系统的并发性能。

## 设计理念

**重要发现**: `~/trident/distributed-deploy/server.py` 已经天然支持并发！
- 使用 `threading.Thread` 为每个客户端连接创建独立线程
- 不需要修改服务器代码
- 只需要客户端并发发送多个查询即可

## 文件说明

- `config.py` - 配置文件（服务器IP、并发级别等）
- `concurrent_benchmark.py` - 主测试脚本
- `README.md` - 本文件

## 使用方法

### 1. 启动服务器（在3台服务器上）

```bash
# 服务器1
cd ~/trident/distributed-deploy
python server.py --server-id 1 --dataset siftsmall

# 服务器2
python server.py --server-id 2 --dataset siftsmall

# 服务器3
python server.py --server-id 3 --dataset siftsmall
```

### 2. 运行并发测试（在客户端机器上）

```bash
cd ~/trident/concurrent-test

# 基本测试（默认并发级别: 1,2,4,8,16）
python concurrent_benchmark.py --dataset siftsmall --queries-per-level 50

# 自定义并发级别
python concurrent_benchmark.py --dataset siftsmall \
  --concurrent-levels "1,2,4,8,16,32,64" \
  --queries-per-level 100

# 快速测试
python concurrent_benchmark.py --dataset siftsmall \
  --concurrent-levels "1,2,4,8" \
  --queries-per-level 10
```

## 输出结果

测试会输出：
1. **实时进度** - 每个并发级别的查询进度
2. **性能指标**:
   - 成功率
   - 吞吐量 (queries/sec)
   - 平均延迟
   - P50/P95/P99延迟
3. **结果文件** - `benchmark_results_*.json`

示例输出：
```
====================================================================================================
性能总结
====================================================================================================
并发级别     成功率      吞吐量(qps)      平均延迟(s)      P95延迟(s)      P99延迟(s)
----------------------------------------------------------------------------------------------------
1            100.0       0.85            1.176           1.200           1.250
2            100.0       1.65            1.212           1.280           1.320
4            100.0       3.12            1.282           1.450           1.520
8            100.0       5.89            1.358           1.680           1.780
16           100.0       10.24           1.562           2.100           2.350
====================================================================================================
```

## 测试目标

验证系统的并发特性：
1. ✅ **线性扩展**: 吞吐量随并发级别线性增长
2. ✅ **稳定延迟**: 延迟保持相对稳定（轻微增长是正常的）
3. ✅ **饱和点**: 找到系统达到最大吞吐量的并发级别
4. ✅ **查询隔离**: 不同查询之间没有干扰（通过稳定的延迟验证）

## 与 distributed-deploy 的关系

- **复用**: 直接使用 `distributed-deploy/client.py` 中的 `DistributedClient` 类
- **简化**: 不需要修改服务器代码
- **专注**: 只实现并发测试逻辑
