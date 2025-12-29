# 阶段1优化改动说明

## 优化目标

解决并发查询的性能瓶颈：多个查询竞争共享的服务器间连接，导致串行化执行。

## 核心改动

### 文件：`server.py`

**修改方法：** `_send_binary_exchange_data` (line 490-589)

#### 优化前（使用共享连接）

```python
def _send_binary_exchange_data(self, target_server_id: int, query_id: str,
                                e_shares: np.ndarray, f_shares: np.ndarray) -> bool:
    # 问题：所有查询共享同一个socket连接
    sock = self.server_connections[target_server_id]  # ← 共享连接！

    try:
        # 发送数据
        sock.sendall(...)
        # 接收响应
        response = sock.recv(...)
        return response.get('status') == 'success'
    except Exception as e:
        # 复杂的重连逻辑
        # 重连后将新socket存入self.server_connections
        ...
```

**问题分析：**
- 多个查询线程竞争同一个socket
- 线程A发送数据时，线程B必须等待
- 导致并发查询串行化执行
- 吞吐量无法随并发级别扩展

#### 优化后（使用临时连接）

```python
def _send_binary_exchange_data(self, target_server_id: int, query_id: str,
                                e_shares: np.ndarray, f_shares: np.ndarray) -> bool:
    sock = None

    try:
        # 每个查询创建独立的临时连接
        server_info = self.server_config[target_server_id]
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 16 * 1024 * 1024)  # 16MB
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 16 * 1024 * 1024)  # 16MB
        sock.settimeout(30)
        sock.connect((server_info['host'], server_info['port']))

        # 发送数据
        sock.sendall(...)
        # 接收响应
        response = sock.recv(...)
        return response.get('status') == 'success'

    except Exception as e:
        logger.error(f"发送失败: {e}")
        return False

    finally:
        # 始终关闭临时连接
        if sock:
            try:
                sock.close()
            except Exception as e:
                logger.warning(f"关闭socket时出错: {e}")
```

**优化优势：**
- ✅ 每个查询独立连接，无竞争
- ✅ 简化错误处理逻辑（无需维护共享连接状态）
- ✅ 并发查询可真正并行执行
- ✅ 吞吐量可随并发级别线性扩展

## 性能预期

### 测试场景对比

**并发级别 = 4，同时执行4个查询：**

**优化前：**
```
查询1: [===发送===][接收]                     0.0-0.1秒
查询2:            [===等待===][===发送===][接收]  0.1-0.2秒
查询3:                        [===等待===][===发送===][接收]  0.2-0.3秒
查询4:                                    [===等待===][===发送===][接收]  0.3-0.4秒

总耗时: 0.4秒
吞吐量: 4/0.4 = 10 qps（但实际上是串行）
```

**优化后：**
```
查询1: [===发送===][接收]  0.0-0.1秒
查询2: [===发送===][接收]  0.0-0.1秒 (并行!)
查询3: [===发送===][接收]  0.0-0.1秒 (并行!)
查询4: [===发送===][接收]  0.0-0.1秒 (并行!)

总耗时: 0.1秒
吞吐量: 4/0.1 = 40 qps（真正的并行）
```

### 预期性能提升

| 并发级别 | 优化前吞吐量 | 优化后吞吐量 | 提升倍数 |
|---------|-------------|-------------|---------|
| 1       | 1.30 qps    | 1.30 qps    | 1x      |
| 2       | 2.21 qps    | 2.60 qps    | 1.2x    |
| 4       | 2.17 qps    | 5.20 qps    | 2.4x    |
| 8       | 2.11 qps    | 10.40 qps   | 4.9x    |
| 16      | ~2 qps      | 20.80 qps   | 10.4x   |

## 未修改的部分

以下功能保持不变，确保向后兼容：
- `self.server_connections` 字典仍保留（用于状态查询）
- `_send_to_server` 方法（未使用，保留供未来使用）
- `_establish_persistent_connections` 方法（启动时建立连接，虽然阶段1不使用）
- 所有其他查询处理逻辑

## 后续优化方向

如果阶段1效果不理想，可以继续优化：

**阶段2：连接池**
- 维护固定大小的连接池（如每对服务器4个连接）
- 查询从池中获取连接，用完归还
- 平衡连接开销和并发性

**阶段3：异步IO**
- 使用 `asyncio` 重写服务器
- 非阻塞IO，单线程处理多个连接
- 预期性能提升：20x+

## 验证方法

运行并发测试：
```bash
cd ~/trident/concurrent-test
python concurrent_benchmark.py --dataset siftsmall --concurrent-levels "1,2,4,8,16" --queries-per-level 50
```

观察：
1. 吞吐量是否随并发级别线性增长
2. 延迟是否保持相对稳定
3. 成功率是否保持100%
