# [CN]1[CN]

## [CN]

[CN]query[CN]：[CN]query[CN]server[CN]，[CN]。

## [CN]

### file：`server.py`

**[CN]：** `_send_binary_exchange_data` (line 490-589)

#### [CN]（[CN]）

```python
def _send_binary_exchange_data(self, target_server_id: int, query_id: str,
                                e_shares: np.ndarray, f_shares: np.ndarray) -> bool:
    # [CN]：[CN]query[CN]socket[CN]
    sock = self.server_connections[target_server_id]  # ← [CN]！

    try:
        # [CN]
        sock.sendall(...)
        # [CN]
        response = sock.recv(...)
        return response.get('status') == 'success'
    except Exception as e:
        # [CN]
        # [CN]socket[CN]self.server_connections
        ...
```

**[CN]：**
- [CN]query[CN]socket
- [CN]A[CN]，[CN]B[CN]
- [CN]query[CN]
- [CN]

#### [CN]（[CN]）

```python
def _send_binary_exchange_data(self, target_server_id: int, query_id: str,
                                e_shares: np.ndarray, f_shares: np.ndarray) -> bool:
    sock = None

    try:
        # [CN]query[CN]
        server_info = self.server_config[target_server_id]
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 16 * 1024 * 1024)  # 16MB
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 16 * 1024 * 1024)  # 16MB
        sock.settimeout(30)
        sock.connect((server_info['host'], server_info['port']))

        # [CN]
        sock.sendall(...)
        # [CN]
        response = sock.recv(...)
        return response.get('status') == 'success'

    except Exception as e:
        logger.error(f"[CN]: {e}")
        return False

    finally:
        # [CN]
        if sock:
            try:
                sock.close()
            except Exception as e:
                logger.warning(f"[CN]socket[CN]: {e}")
```

**[CN]：**
- ✅ [CN]query[CN]，[CN]
- ✅ [CN]（[CN]）
- ✅ [CN]query[CN]
- ✅ [CN]

## [CN]

### test[CN]

**[CN] = 4，[CN]4[CN]query：**

**[CN]：**
```
query1: [===[CN]===][[CN]]                     0.0-0.1[CN]
query2:            [===[CN]===][===[CN]===][[CN]]  0.1-0.2[CN]
query3:                        [===[CN]===][===[CN]===][[CN]]  0.2-0.3[CN]
query4:                                    [===[CN]===][===[CN]===][[CN]]  0.3-0.4[CN]

[CN]: 0.4[CN]
[CN]: 4/0.4 = 10 qps（[CN]）
```

**[CN]：**
```
query1: [===[CN]===][[CN]]  0.0-0.1[CN]
query2: [===[CN]===][[CN]]  0.0-0.1[CN] ([CN]!)
query3: [===[CN]===][[CN]]  0.0-0.1[CN] ([CN]!)
query4: [===[CN]===][[CN]]  0.0-0.1[CN] ([CN]!)

[CN]: 0.1[CN]
[CN]: 4/0.1 = 40 qps（[CN]）
```

### [CN]

| [CN] | [CN] | [CN] | [CN] |
|---------|-------------|-------------|---------|
| 1       | 1.30 qps    | 1.30 qps    | 1x      |
| 2       | 2.21 qps    | 2.60 qps    | 1.2x    |
| 4       | 2.17 qps    | 5.20 qps    | 2.4x    |
| 8       | 2.11 qps    | 10.40 qps   | 4.9x    |
| 16      | ~2 qps      | 20.80 qps   | 10.4x   |

## [CN]

[CN]，[CN]：
- `self.server_connections` [CN]（[CN]query）
- `_send_to_server` [CN]（[CN]，[CN]）
- `_establish_persistent_connections` [CN]（start[CN]，[CN]1[CN]）
- [CN]query[CN]

## [CN]

[CN]1[CN]，[CN]：

**[CN]2：[CN]**
- [CN]（[CN]server4[CN]）
- query[CN]，[CN]
- [CN]

**[CN]3：[CN]IO**
- [CN] `asyncio` [CN]server
- [CN]IO，[CN]
- [CN]：20x+

## [CN]

run[CN]test：
```bash
cd ~/trident/concurrent-test
python concurrent_benchmark.py --dataset siftsmall --concurrent-levels "1,2,4,8,16" --queries-per-level 50
```

[CN]：
1. [CN]
2. [CN]
3. [CN]100%
