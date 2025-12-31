# stage1optimization[CN]description

## optimization objective

resolveconcurrentquery[CN]performancebottleneck:[CN]query[CN]share[CN]server[CN]connect,[CN]serializableexecute.

## [CN]

### file:`server.py`

**modifymethod:** `_send_binary_exchange_data` (line 490-589)

#### optimizationbefore(usageshareconnect)

```python
def _send_binary_exchange_data(self, target_server_id: int, query_id: str,
                                e_shares: np.ndarray, f_shares: np.ndarray) -> bool:
    # issue:[CN]queryshare[CN]socketconnect
    sock = self.server_connections[target_server_id]  # ← shareconnect!

    try:
        # senddata
        sock.sendall(...)
        # receiveresponse
        response = sock.recv(...)
        return response.get('status') == 'success'
    except Exception as e:
        # complex[CN]
        # [CN]after[CN]socket[CN]self.server_connections
        ...
```

**issueanalysis:**
- [CN]querythread[CN]socket
- threadAsenddatahour,threadB[CN]wait
- [CN]concurrentqueryserializableexecute
- throughput[CN]concurrent[CN]extension

#### optimizationafter(usage[CN]hourconnect)

```python
def _send_binary_exchange_data(self, target_server_id: int, query_id: str,
                                e_shares: np.ndarray, f_shares: np.ndarray) -> bool:
    sock = None

    try:
        # [CN]querycreateindependent[CN]hourconnect
        server_info = self.server_config[target_server_id]
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 16 * 1024 * 1024)  # 16MB
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 16 * 1024 * 1024)  # 16MB
        sock.settimeout(30)
        sock.connect((server_info['host'], server_info['port']))

        # senddata
        sock.sendall(...)
        # receiveresponse
        response = sock.recv(...)
        return response.get('status') == 'success'

    except Exception as e:
        logger.error(f"sendfailure: {e}")
        return False

    finally:
        # [CN]close[CN]hourconnect
        if sock:
            try:
                sock.close()
            except Exception as e:
                logger.warning(f"closesockethour[CN]: {e}")
```

**optimization[CN]:**
- ✅ [CN]queryindependentconnect,[CN]
- ✅ simplificationerrorhandle[CN]([CN]shareconnectstate)
- ✅ concurrentquery[CN]truepositiveparallelexecute
- ✅ throughput[CN]concurrent[CN]extension

## performance[CN]

### testscene[CN]

**concurrent[CN] = 4,[CN]hourexecute4[CN]query:**

**optimizationbefore:**
```
query1: [===send===][receive]                     0.0-0.1second
query2:            [===wait===][===send===][receive]  0.1-0.2second
query3:                        [===wait===][===send===][receive]  0.2-0.3second
query4:                                    [===wait===][===send===][receive]  0.3-0.4second

[CN]hour: 0.4second
throughput: 4/0.4 = 10 qps([CN]onyesserial)
```

**optimizationafter:**
```
query1: [===send===][receive]  0.0-0.1second
query2: [===send===][receive]  0.0-0.1second (parallel!)
query3: [===send===][receive]  0.0-0.1second (parallel!)
query4: [===send===][receive]  0.0-0.1second (parallel!)

[CN]hour: 0.1second
throughput: 4/0.1 = 40 qps(truepositive[CN]parallel)
```

### [CN]performance[CN]

| concurrent[CN] | optimizationbeforethroughput | optimizationafterthroughput | [CN]multiple |
|---------|-------------|-------------|---------|
| 1       | 1.30 qps    | 1.30 qps    | 1x      |
| 2       | 2.21 qps    | 2.60 qps    | 1.2x    |
| 4       | 2.17 qps    | 5.20 qps    | 2.4x    |
| 8       | 2.11 qps    | 10.40 qps   | 4.9x    |
| 16      | ~2 qps      | 20.80 qps   | 10.4x   |

## [CN]modify[CN]minute

[CN]underfunctionality[CN]invariance,[CN]backward compatible:
- `self.server_connections` dictionary[CN]retention([CN]statequery)
- `_send_to_server` method([CN]usage,retention[CN]usage)
- `_establish_persistent_connections` method(starthour[CN]connect,[CN]stage1[CN]usage)
- [CN]queryhandle[CN]

## after[CN]optimization[CN]

[CN]stage1[CN],[CN]continueoptimization:

**stage2:connection pool**
- [CN]size[CN]connection pool([CN]server4[CN]connect)
- query[CN]poolinfetchconnect,[CN]
- balancedconnectoverhead[CN]concurrent[CN]

**stage3:asynchronousIO**
- usage `asyncio` overrideserver
- non-blockingIO,single-threadedhandle[CN]connect
- [CN]performance[CN]:20x+

## verificationmethod

runconcurrenttest:
```bash
cd ~/trident/concurrent-test
python concurrent_benchmark.py --dataset siftsmall --concurrent-levels "1,2,4,8,16" --queries-per-level 50
```

[CN]:
1. throughputyesno[CN]concurrent[CN]
2. lateyesno[CN]stable
3. success[CN]yesno[CN]100%
