# [CN]query[CN] (Distributed-NL)

[CN]MPC[CN]，[CN]queryvector[CN]K[CN]index。

## [CN]

### [CN]distributed-deploy[CN]

| [CN] | distributed-deploy | distributed-nl |
|------|-------------------|----------------|
| **[CN]** | queryvector[CN] | query[CN] |
| **input** | [CN]index | query[CN]index |
| **output** | [CN]vector[CN] | K[CN]index |
| **[CN]** | 8001-8003 | 9001-9003 |
| **[CN]** | nodes_shares.npy | neighbors_shares.npy |

## [CN]configure

- **client → server**: [CN]IP[CN]
  - Server1: `192.168.1.101:9001`
  - Server2: `192.168.1.102:9002`
  - Server3: `192.168.1.103:9003`

- **server ↔ server**: [CN]IP[CN]
  - Server1: `10.0.1.101:9001`
  - Server2: `10.0.1.102:9002`
  - Server3: `10.0.1.103:9003`

## [CN]

### 1. startserver

[CN]server[CN]run：

```bash
# server1
python server.py --server-id 1 --dataset siftsmall --vdpf-processes 4

# server2
python server.py --server-id 2 --dataset siftsmall --vdpf-processes 4

# server3
python server.py --server-id 3 --dataset siftsmall --vdpf-processes 4
```

### 2. runclienttest

```bash
# [CN]test
python client.py --dataset siftsmall --num-queries 10

# test[CN]dataset
python client.py --dataset laion --num-queries 5

# [CN]server[CN]
python client.py --status-only
```

## deploy[CN]

### [CN]server

```bash
./deploy.sh
```

### [CN]server[CN]start[CN]

```bash
# start[CN]server
./start-servers.sh

# stop[CN]server
./stop-servers.sh
```

## [CN]

### 1. TCPparameters[CN]（[CN]server[CN]）

```bash
sudo sysctl -w net.core.rmem_max=268435456
sudo sysctl -w net.core.wmem_max=268435456
sudo sysctl -w net.ipv4.tcp_rmem="4096 87380 268435456"
sudo sysctl -w net.ipv4.tcp_wmem="4096 65536 268435456"
```

### 2. [CN]

[CN]CPU[CN]：
```bash
python server.py --server-id 1 --vdpf-processes 32  # [CN]64[CN]
```

## dataset[CN]

| dataset | [CN](K) | [CN] | [CN] |
|--------|-----------|--------|------|
| siftsmall | 100 | 10,000 | ✅ [CN] |
| laion | 36 | 100,000 | ✅ [CN] |
| nfcorpus | 10 | 3,633 | ✅ [CN] |
| tripclick | 36 | 1.5M | 🟡 [CN] |

## [CN]

[CN]：
- [CN]
- [CN]
- [CN]query[CN]
- [CN]

## test[CN]

testresult[CN] `nl_result.md`，[CN]：
- [CN]query[CN]
- [CN]
- [CN]
- [CN]

## [CN]

### [CN]
- [CN]9001-9003[CN]
- [CN]server[CN]
- [CN]

### [CN]
- [CN]MSG_WAITALL[CN]，[CN]
- [CN]，[CN]

### [CN]
- [CN]neighbors_shares.npy[CN]generated
- [CN]groundtruth[CN]

## [CN]

### [CN]serverconfigure

[CN] `custom_servers.json`:
```json
{
  "1": {"host": "192.168.1.101", "port": 9001},
  "2": {"host": "192.168.1.102", "port": 9002},
  "3": {"host": "192.168.1.103", "port": 9003}
}
```

[CN]configure：
```bash
python client.py --config custom_servers.json
```

## [CN]vectorquery[CN]

[CN]vector[CN]：
1. [CN] `distributed-nl` queryK[CN]index
2. [CN] `distributed-deploy` [CN]vector[CN]
3. [CN]client[CN]

[CN]。