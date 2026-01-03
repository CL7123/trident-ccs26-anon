# Phase 1 Optimization Deployment Guide

## Optimization Content

**Phase 1: Replace shared connections with temporary connections**
- Create independent socket connections for each query to other servers
- Avoid multiple thread contention on the same socket (eliminate concurrency bottleneck)
- Close connections immediately after sending
- Expected performance improvement: 4-5x throughput increase (from ~2 qps to ~8-10 qps)

## Deployment Steps

### 1. Deploy optimized code on 3 servers

**Server 1 (192.168.1.103):**
```bash
# SSH login
ssh -i ~/trident/your-key.pem ubuntu@192.168.1.103

# Stop old server (if running)
pkill -f "python.*server.py"

# Backup original code
cd ~/trident
cp -r distributed-deploy distributed-deploy-backup-$(date +%Y%m%d_%H%M%S)

# Sync optimized code (run on local machine)
# After exiting SSH, run on local:
scp -i ~/trident/your-key.pem ~/trident/distributed-deploy-optimized/server.py ubuntu@192.168.1.103:~/trident/distributed-deploy/

# Re-login and start optimized server
ssh -i ~/trident/your-key.pem ubuntu@192.168.1.103
cd ~/trident/distributed-deploy
source venv/bin/activate
nohup python server.py --server-id 1 --dataset siftsmall > server1.log 2>&1 &

# Check startup status
tail -f server1.log
```

**Server 2 (192.168.1.101):**
```bash
ssh -i ~/trident/your-key.pem ubuntu@192.168.1.101
pkill -f "python.*server.py"

cd ~/trident
cp -r distributed-deploy distributed-deploy-backup-$(date +%Y%m%d_%H%M%S)

# Run on local:
scp -i ~/trident/your-key.pem ~/trident/distributed-deploy-optimized/server.py ubuntu@192.168.1.101:~/trident/distributed-deploy/

# Re-login and start
ssh -i ~/trident/your-key.pem ubuntu@192.168.1.101
cd ~/trident/distributed-deploy
source venv/bin/activate
nohup python server.py --server-id 2 --dataset siftsmall > server2.log 2>&1 &
tail -f server2.log
```

**Server 3 (192.168.1.103):**
```bash
ssh -i ~/trident/your-key.pem ubuntu@192.168.1.103
pkill -f "python.*server.py"

cd ~/trident
cp -r distributed-deploy distributed-deploy-backup-$(date +%Y%m%d_%H%M%S)

# Run on local:
scp -i ~/trident/your-key.pem ~/trident/distributed-deploy-optimized/server.py ubuntu@192.168.1.103:~/trident/distributed-deploy/

# Re-login and start
ssh -i ~/trident/your-key.pem ubuntu@192.168.1.103
cd ~/trident/distributed-deploy
source venv/bin/activate
nohup python server.py --server-id 3 --dataset siftsmall > server3.log 2>&1 &
tail -f server3.log
```

### 2. Run concurrent performance tests

```bash
cd ~/trident/concurrent-test
source /home/ubuntu/venv/bin/activate

# Run test
python concurrent_benchmark.py \
  --dataset siftsmall \
  --concurrent-levels "1,2,4,8,16" \
  --queries-per-level 50
```

### 3. Expected results comparison

**Before optimization (shared connections):**
```
Concurrency Level   Success Rate   Throughput (qps)   Avg Latency (s)   P95 Latency (s)   P99 Latency (s)
1                   100.0          1.30               0.769             0.850             0.900
2                   100.0          2.21               0.904             1.020             1.080
4                   100.0          2.17               1.842             2.150             2.250
8                   100.0          2.11               3.790             4.200             4.450
```

**After optimization (temporary connections) expected:**
```
Concurrency Level   Success Rate   Throughput (qps)   Avg Latency (s)   P95 Latency (s)   P99 Latency (s)
1                   100.0          1.30               0.769             0.850             0.900
2                   100.0          2.50               0.800             0.880             0.920
4                   100.0          4.80               0.833             0.950             1.000
8                   100.0          8.50               0.941             1.100             1.180
16                  100.0          12.00              1.333             1.600             1.750
```

Main improvements:
- ✅ Throughput scales nearly linearly (no longer limited by shared connections)
- ✅ Latency remains stable (slight increase)
- ✅ At concurrency level 8, throughput increases from 2.11 to ~8.5 (4x improvement)

### 4. Rollback if test fails

```bash
# On each server
pkill -f "python.*server.py"
cd ~/trident
# Find latest backup
ls -lt | grep distributed-deploy-backup
# Restore backup (replace timestamp below)
cp distributed-deploy-backup-TIMESTAMP/server.py distributed-deploy/
# Restart server
cd distributed-deploy
source venv/bin/activate
nohup python server.py --server-id [1/2/3] --dataset siftsmall > server.log 2>&1 &
```

## Technical Details

### Key Changes

In `server.py:490` `_send_binary_exchange_data` method:

**Before optimization (problematic):**
```python
# Use shared connections
sock = self.server_connections[target_server_id]
sock.sendall(...)  # Multiple threads compete for same socket
```

**After optimization (Phase 1):**
```python
try:
    # Create temporary connection for each query
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((server_info['host'], server_info['port']))
    sock.sendall(...)
    return response.get('status') == 'success'
finally:
    # Always close connection
    if sock:
        sock.close()
```

### Future Optimization Directions

If Phase 1 doesn't achieve desired results, further improvements possible:
- **Phase 2: Connection pooling** - Balance connection overhead and concurrency (expected 8-16x improvement)
- **Phase 3: Async IO** - Use asyncio for non-blocking IO (expected 20x+ improvement)
