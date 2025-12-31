# stage1optimizationdeployment[CN]

## optimizationcontent

**stage1:[CN]hourconnectreplaceshareconnect**
- [CN]querycreateindependent[CN]socketconnect[CN]server
- [CN]multi-threaded[CN]socket([CN]concurrentbottleneck)
- sendcompleteafterimmediatecloseconnect
- [CN]performance[CN]:4-5[CN]throughput([CN]~2 qps[CN]~8-10 qps)

## deployment[CN]

### 1. [CN]3[CN]serverondeploymentoptimizationafter[CN]code

**server1 (192.168.1.103):**
```bash
# SSH[CN]
ssh -i ~/trident/your-key.pem ubuntu@192.168.1.103

# stop[CN]server([CN]run)
pkill -f "python.*server.py"

# backup[CN]code
cd ~/trident
cp -r distributed-deploy distributed-deploy-backup-$(date +%Y%m%d_%H%M%S)

# synchronousoptimizationafter[CN]code([CN]local[CN]run)
# exitSSHafter[CN]localrun:
scp -i ~/trident/your-key.pem ~/trident/distributed-deploy-optimized/server.py ubuntu@192.168.1.103:~/trident/distributed-deploy/

# [CN]startoptimizationafter[CN]server
ssh -i ~/trident/your-key.pem ubuntu@192.168.1.103
cd ~/trident/distributed-deploy
source venv/bin/activate
nohup python server.py --server-id 1 --dataset siftsmall > server1.log 2>&1 &

# checkstartstate
tail -f server1.log
```

**server2 (192.168.1.101):**
```bash
ssh -i ~/trident/your-key.pem ubuntu@192.168.1.101
pkill -f "python.*server.py"

cd ~/trident
cp -r distributed-deploy distributed-deploy-backup-$(date +%Y%m%d_%H%M%S)

# [CN]localrun:
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

# [CN]localrun:
scp -i ~/trident/your-key.pem ~/trident/distributed-deploy-optimized/server.py ubuntu@192.168.1.103:~/trident/distributed-deploy/

# [CN]start
ssh -i ~/trident/your-key.pem ubuntu@192.168.1.103
cd ~/trident/distributed-deploy
source venv/bin/activate
nohup python server.py --server-id 3 --dataset siftsmall > server3.log 2>&1 &
tail -f server3.log
```

### 2. runconcurrentperformance test

```bash
cd ~/trident/concurrent-test
source /home/ubuntu/venv/bin/activate

# runtest
python concurrent_benchmark.py \
  --dataset siftsmall \
  --concurrent-levels "1,2,4,8,16" \
  --queries-per-level 50
```

### 3. [CN]result[CN]

**optimizationbefore(shareconnect):**
```
concurrent[CN]     success[CN]      throughput(qps)      averagelate(s)      P95late(s)      P99late(s)
1            100.0       1.30            0.769           0.850           0.900
2            100.0       2.21            0.904           1.020           1.080
4            100.0       2.17            1.842           2.150           2.250
8            100.0       2.11            3.790           4.200           4.450
```

**optimizationafter([CN]hourconnect)[CN]:**
```
concurrent[CN]     success[CN]      throughput(qps)      averagelate(s)      P95late(s)      P99late(s)
1            100.0       1.30            0.769           0.850           0.900
2            100.0       2.50            0.800           0.880           0.920
4            100.0       4.80            0.833           0.950           1.000
8            100.0       8.50            0.941           1.100           1.180
16           100.0       12.00           1.333           1.600           1.750
```

[CN]improvement:
- ✅ throughput[CN]extension([CN]shareconnectlimit)
- ✅ late[CN]stable([CN])
- ✅ concurrent[CN]8hourthroughput[CN]2.11[CN]~8.5(4[CN])

### 4. [CN]testfailure[CN]rollback

```bash
# [CN]serveron
pkill -f "python.*server.py"
cd ~/trident
# [CN]backup
ls -lt | grep distributed-deploy-backup
# recoverybackup(replaceunder[CN]timestamp)
cp distributed-deploy-backup-TIMESTAMP/server.py distributed-deploy/
# [CN]server
cd distributed-deploy
source venv/bin/activate
nohup python server.py --server-id [1/2/3] --dataset siftsmall > server.log 2>&1 &
```

## [CN]

### [CN]keymodify

[CN] `server.py:490` [CN] `_send_binary_exchange_data` method:

**optimizationbefore([CN]issue):**
```python
# usageshareconnect
sock = self.server_connections[target_server_id]
sock.sendall(...)  # [CN]thread[CN]socket
```

**optimizationafter(stage1):**
```python
try:
    # [CN]querycreate[CN]hourconnect
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((server_info['host'], server_info['port']))
    sock.sendall(...)
    return response.get('status') == 'success'
finally:
    # [CN]closeconnect
    if sock:
        sock.close()
```

### after[CN]optimization[CN]

[CN]stage1[CN],[CN]continue:
- **stage2:connection pool** - balancedconnectoverhead[CN]concurrent[CN]([CN]8-16x[CN])
- **stage3:asynchronousIO** - usageasyncioimplementationnon-blockingIO([CN]20x+[CN])
