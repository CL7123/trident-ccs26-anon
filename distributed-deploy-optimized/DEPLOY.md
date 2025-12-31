# [CN]1[CN]deploy[CN]

## [CN]

**[CN]1：[CN]**
- [CN]query[CN]socket[CN]server
- [CN]socket（[CN]）
- [CN]
- [CN]：4-5[CN]（[CN]~2 qps[CN]~8-10 qps）

## deploy[CN]

### 1. [CN]3[CN]server[CN]deploy[CN]

**server1 (192.168.1.103):**
```bash
# SSH[CN]
ssh -i ~/trident/your-key.pem ubuntu@192.168.1.103

# stop[CN]server（[CN]run）
pkill -f "python.*server.py"

# [CN]
cd ~/trident
cp -r distributed-deploy distributed-deploy-backup-$(date +%Y%m%d_%H%M%S)

# [CN]（[CN]run）
# [CN]SSH[CN]run：
scp -i ~/trident/your-key.pem ~/trident/distributed-deploy-optimized/server.py ubuntu@192.168.1.103:~/trident/distributed-deploy/

# [CN]start[CN]server
ssh -i ~/trident/your-key.pem ubuntu@192.168.1.103
cd ~/trident/distributed-deploy
source venv/bin/activate
nohup python server.py --server-id 1 --dataset siftsmall > server1.log 2>&1 &

# [CN]start[CN]
tail -f server1.log
```

**server2 (192.168.1.101):**
```bash
ssh -i ~/trident/your-key.pem ubuntu@192.168.1.101
pkill -f "python.*server.py"

cd ~/trident
cp -r distributed-deploy distributed-deploy-backup-$(date +%Y%m%d_%H%M%S)

# [CN]run：
scp -i ~/trident/your-key.pem ~/trident/distributed-deploy-optimized/server.py ubuntu@192.168.1.101:~/trident/distributed-deploy/

# [CN]start
ssh -i ~/trident/your-key.pem ubuntu@192.168.1.101
cd ~/trident/distributed-deploy
source venv/bin/activate
nohup python server.py --server-id 2 --dataset siftsmall > server2.log 2>&1 &
tail -f server2.log
```

**server3 (192.168.1.103):**
```bash
ssh -i ~/trident/your-key.pem ubuntu@192.168.1.103
pkill -f "python.*server.py"

cd ~/trident
cp -r distributed-deploy distributed-deploy-backup-$(date +%Y%m%d_%H%M%S)

# [CN]run：
scp -i ~/trident/your-key.pem ~/trident/distributed-deploy-optimized/server.py ubuntu@192.168.1.103:~/trident/distributed-deploy/

# [CN]start
ssh -i ~/trident/your-key.pem ubuntu@192.168.1.103
cd ~/trident/distributed-deploy
source venv/bin/activate
nohup python server.py --server-id 3 --dataset siftsmall > server3.log 2>&1 &
tail -f server3.log
```

### 2. run[CN]test

```bash
cd ~/trident/concurrent-test
source ~/venv/bin/activate

# runtest
python concurrent_benchmark.py \
  --dataset siftsmall \
  --concurrent-levels "1,2,4,8,16" \
  --queries-per-level 50
```

### 3. [CN]result[CN]

**[CN]（[CN]）：**
```
[CN]     [CN]      [CN](qps)      [CN](s)      P95[CN](s)      P99[CN](s)
1            100.0       1.30            0.769           0.850           0.900
2            100.0       2.21            0.904           1.020           1.080
4            100.0       2.17            1.842           2.150           2.250
8            100.0       2.11            3.790           4.200           4.450
```

**[CN]（[CN]）[CN]：**
```
[CN]     [CN]      [CN](qps)      [CN](s)      P95[CN](s)      P99[CN](s)
1            100.0       1.30            0.769           0.850           0.900
2            100.0       2.50            0.800           0.880           0.920
4            100.0       4.80            0.833           0.950           1.000
8            100.0       8.50            0.941           1.100           1.180
16           100.0       12.00           1.333           1.600           1.750
```

[CN]：
- ✅ [CN]（[CN]）
- ✅ [CN]（[CN]）
- ✅ [CN]8[CN]2.11[CN]~8.5（4[CN]）

### 4. [CN]test[CN]

```bash
# [CN]server[CN]
pkill -f "python.*server.py"
cd ~/trident
# [CN]
ls -lt | grep distributed-deploy-backup
# [CN]（[CN]）
cp distributed-deploy-backup-TIMESTAMP/server.py distributed-deploy/
# [CN]server
cd distributed-deploy
source venv/bin/activate
nohup python server.py --server-id [1/2/3] --dataset siftsmall > server.log 2>&1 &
```

## [CN]

### [CN]

[CN] `server.py:490` [CN] `_send_binary_exchange_data` [CN]：

**[CN]（[CN]）：**
```python
# [CN]
sock = self.server_connections[target_server_id]
sock.sendall(...)  # [CN]socket
```

**[CN]（[CN]1）：**
```python
try:
    # [CN]query[CN]
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((server_info['host'], server_info['port']))
    sock.sendall(...)
    return response.get('status') == 'success'
finally:
    # [CN]
    if sock:
        sock.close()
```

### [CN]

[CN]1[CN]，[CN]：
- **[CN]2：[CN]** - [CN]（[CN]8-16x[CN]）
- **[CN]3：[CN]IO** - [CN]asyncio[CN]IO（[CN]20x+[CN]）
