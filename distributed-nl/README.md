# distributedneighbor listquerysystem (Distributed-NL)

[CN]yes[CN]MPC[CN]distributedneighbor list[CN]system,[CN]security[CN]queryvector[CN]K[CN]neighborindex.

## systemarchitecture

### [CN]distributed-deploy[CN]

| attribute | distributed-deploy | distributed-nl |
|------|-------------------|----------------|
| **functionality** | queryvectorcontent | queryneighbor list |
| **input** | nodeindex | querynodeindex |
| **output** | [CN]vectorvalue | K[CN]neighborindex |
| **port** | 8001-8003 | 9001-9003 |
| **data** | nodes_shares.npy | neighbors_shares.npy |

## networkconfiguration

- **client → server**: usage[CN]IP[CN]
  - Server1: `192.168.1.101:9001`
  - Server2: `192.168.1.102:9002`
  - Server3: `192.168.1.103:9003`

- **server ↔ server**: usage[CN]IP[CN]
  - Server1: `10.0.1.101:9001`
  - Server2: `10.0.1.102:9002`
  - Server3: `10.0.1.103:9003`

## faststart

### 1. startserver

[CN]serveronrun:

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
# basictest
python client.py --dataset siftsmall --num-queries 10

# test[CN]dataset
python client.py --dataset laion --num-queries 5

# [CN]serverstate
python client.py --status-only
```

## deployment[CN]

### synchronous[CN]server

```bash
./deploy.sh
```

### [CN]serveronstartservice

```bash
# start[CN]neighbor listserver
./start-servers.sh

# stop[CN]server
./stop-servers.sh
```

## performanceoptimization

### 1. TCPparametersoptimization([CN]serverexecute)

```bash
sudo sysctl -w net.core.rmem_max=268435456
sudo sysctl -w net.core.wmem_max=268435456
sudo sysctl -w net.ipv4.tcp_rmem="4096 87380 268435456"
sudo sysctl -w net.ipv4.tcp_wmem="4096 65536 268435456"
```

### 2. process[CN]optimization

root[CN]CPU[CN]:
```bash
python server.py --server-id 1 --vdpf-processes 32  # [CN]64[CN]
```

## datasetsupport

| dataset | neighbor[CN](K) | node[CN] | state |
|--------|-----------|--------|------|
| siftsmall | 100 | 10,000 | ✅ [CN]support |
| laion | 36 | 100,000 | ✅ [CN]support |
| nfcorpus | 10 | 3,633 | ✅ [CN]support |
| tripclick | 36 | 1.5M | 🟡 [CN]data[CN] |

## monitorfunctionality

system[CN]automatic[CN]display:
- [CN]stage[CN]executetime
- network[CN]size[CN]velocity
- neighbor listqueryaccuracy
- detailed[CN]performancestatistics

## testreport

testresult[CN]automaticsave[CN] `nl_result.md`,package[CN]:
- detailed[CN]querytimefactorization
- network[CN]statistics
- accuracyassess
- performance analysis

## [CN]

### connectfailure
- checksecurity[CN]yesno[CN]9001-9003port
- acknowledgmentserverpositive[CN]listencorrect[CN]port
- verificationnetworkconnected[CN]

### data[CN]error
- system[CN]usageMSG_WAITALLoptimization,[CN]issue
- [CN]issue,checknetworkstable[CN]

### accuracyissue
- [CN]neighbors_shares.npydatacorrectgenerate
- verificationgroundtruthdataformatmatch

## advancedfunctionality

### customserverconfiguration

create `custom_servers.json`:
```json
{
  "1": {"host": "192.168.1.101", "port": 9001},
  "2": {"host": "192.168.1.102", "port": 9002},
  "3": {"host": "192.168.1.103", "port": 9003}
}
```

usagecustomconfiguration:
```bash
python client.py --config custom_servers.json
```

## [CN]vectorquerysystem[CN]

[CN]vectorsearchstream[CN]:
1. usage `distributed-nl` queryK[CN]neighborindex
2. usage `distributed-deploy` [CN]neighbor[CN]vectorcontent
3. [CN]clientcomputationexactsimilarity[CN]sort

[CN]minute[CN]flexibility[CN]performanceoptimizationnull[CN].