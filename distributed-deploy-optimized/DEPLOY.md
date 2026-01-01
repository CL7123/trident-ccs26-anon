# 阶段1优化部署指南

## 优化内容

**阶段1：临时连接替换共享连接**
- 每个查询创建独立的socket连接到其他服务器
- 避免多线程竞争同一socket（消除并发瓶颈）
- 发送完成后立即关闭连接
- 预期性能提升：4-5倍吞吐量（从~2 qps提升到~8-10 qps）

## 部署步骤

### 1. 在3台服务器上部署优化后的代码

**服务器1 (192.168.1.103):**
```bash
# SSH登录
ssh -i ~/trident/your-key.pem ubuntu@192.168.1.103

# 停止旧服务器（如果在运行）
pkill -f "python.*server.py"

# 备份原始代码
cd ~/trident
cp -r distributed-deploy distributed-deploy-backup-$(date +%Y%m%d_%H%M%S)

# 同步优化后的代码（在本地机器运行）
# 退出SSH后在本地运行：
scp -i ~/trident/your-key.pem ~/trident/distributed-deploy-optimized/server.py ubuntu@192.168.1.103:~/trident/distributed-deploy/

# 重新登录并启动优化后的服务器
ssh -i ~/trident/your-key.pem ubuntu@192.168.1.103
cd ~/trident/distributed-deploy
source venv/bin/activate
nohup python server.py --server-id 1 --dataset siftsmall > server1.log 2>&1 &

# 检查启动状态
tail -f server1.log
```

**服务器2 (192.168.1.101):**
```bash
ssh -i ~/trident/your-key.pem ubuntu@192.168.1.101
pkill -f "python.*server.py"

cd ~/trident
cp -r distributed-deploy distributed-deploy-backup-$(date +%Y%m%d_%H%M%S)

# 在本地运行：
scp -i ~/trident/your-key.pem ~/trident/distributed-deploy-optimized/server.py ubuntu@192.168.1.101:~/trident/distributed-deploy/

# 重新登录启动
ssh -i ~/trident/your-key.pem ubuntu@192.168.1.101
cd ~/trident/distributed-deploy
source venv/bin/activate
nohup python server.py --server-id 2 --dataset siftsmall > server2.log 2>&1 &
tail -f server2.log
```

**服务器3 (192.168.1.103):**
```bash
ssh -i ~/trident/your-key.pem ubuntu@192.168.1.103
pkill -f "python.*server.py"

cd ~/trident
cp -r distributed-deploy distributed-deploy-backup-$(date +%Y%m%d_%H%M%S)

# 在本地运行：
scp -i ~/trident/your-key.pem ~/trident/distributed-deploy-optimized/server.py ubuntu@192.168.1.103:~/trident/distributed-deploy/

# 重新登录启动
ssh -i ~/trident/your-key.pem ubuntu@192.168.1.103
cd ~/trident/distributed-deploy
source venv/bin/activate
nohup python server.py --server-id 3 --dataset siftsmall > server3.log 2>&1 &
tail -f server3.log
```

### 2. 运行并发性能测试

```bash
cd ~/trident/concurrent-test
source /home/ubuntu/venv/bin/activate

# 运行测试
python concurrent_benchmark.py \
  --dataset siftsmall \
  --concurrent-levels "1,2,4,8,16" \
  --queries-per-level 50
```

### 3. 预期结果对比

**优化前（共享连接）：**
```
并发级别     成功率      吞吐量(qps)      平均延迟(s)      P95延迟(s)      P99延迟(s)
1            100.0       1.30            0.769           0.850           0.900
2            100.0       2.21            0.904           1.020           1.080
4            100.0       2.17            1.842           2.150           2.250
8            100.0       2.11            3.790           4.200           4.450
```

**优化后（临时连接）预期：**
```
并发级别     成功率      吞吐量(qps)      平均延迟(s)      P95延迟(s)      P99延迟(s)
1            100.0       1.30            0.769           0.850           0.900
2            100.0       2.50            0.800           0.880           0.920
4            100.0       4.80            0.833           0.950           1.000
8            100.0       8.50            0.941           1.100           1.180
16           100.0       12.00           1.333           1.600           1.750
```

主要改进：
- ✅ 吞吐量接近线性扩展（不再受共享连接限制）
- ✅ 延迟保持稳定（轻微增长）
- ✅ 并发级别8时吞吐量从2.11提升到~8.5（4倍提升）

### 4. 如果测试失败需要回滚

```bash
# 在每台服务器上
pkill -f "python.*server.py"
cd ~/trident
# 找到最新的备份
ls -lt | grep distributed-deploy-backup
# 恢复备份（替换下面的时间戳）
cp distributed-deploy-backup-TIMESTAMP/server.py distributed-deploy/
# 重启服务器
cd distributed-deploy
source venv/bin/activate
nohup python server.py --server-id [1/2/3] --dataset siftsmall > server.log 2>&1 &
```

## 技术细节

### 关键修改

在 `server.py:490` 的 `_send_binary_exchange_data` 方法：

**优化前（有问题）：**
```python
# 使用共享连接
sock = self.server_connections[target_server_id]
sock.sendall(...)  # 多个线程竞争同一socket
```

**优化后（阶段1）：**
```python
try:
    # 每个查询创建临时连接
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((server_info['host'], server_info['port']))
    sock.sendall(...)
    return response.get('status') == 'success'
finally:
    # 始终关闭连接
    if sock:
        sock.close()
```

### 后续优化方向

如果阶段1效果不理想，还可以继续：
- **阶段2：连接池** - 平衡连接开销和并发性（预期8-16x提升）
- **阶段3：异步IO** - 使用asyncio实现非阻塞IO（预期20x+提升）
