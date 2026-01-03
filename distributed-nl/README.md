# Distributed Neighbor List Query System (Distributed-NL)

This is an MPC-based distributed neighbor list retrieval system specifically designed for securely querying K-nearest neighbor indices of vectors.

## System Architecture

### Differences from distributed-deploy

| Feature | distributed-deploy | distributed-nl |
|---------|-------------------|----------------|
| **Function** | Query vector content | Query neighbor list |
| **Input** | Node index | Query node index |
| **Output** | Complete vector values | K neighbor indices |
| **Port** | 8001-8003 | 9001-9003 |
| **Data** | nodes_shares.npy | neighbors_shares.npy |

## Network Configuration

- **Client → Server**: Using public IP communication
  - Server1: `192.168.1.101:9001`
  - Server2: `192.168.1.102:9002`
  - Server3: `192.168.1.103:9003`

- **Server ↔ Server**: Using private IP communication
  - Server1: `10.0.1.101:9001`
  - Server2: `10.0.1.102:9002`
  - Server3: `10.0.1.103:9003`

## Quick Start

### 1. Start Servers

Run on each server:

```bash
# Server 1
python server.py --server-id 1 --dataset siftsmall --vdpf-processes 4

# Server 2
python server.py --server-id 2 --dataset siftsmall --vdpf-processes 4

# Server 3
python server.py --server-id 3 --dataset siftsmall --vdpf-processes 4
```

### 2. Run Client Tests

```bash
# Basic test
python client.py --dataset siftsmall --num-queries 10

# Test with other datasets
python client.py --dataset laion --num-queries 5

# Check server status only
python client.py --status-only
```

## Deployment Scripts

### Sync to All Servers

```bash
./deploy.sh
```

### Start Services on All Servers

```bash
# Start all neighbor list servers
./start-servers.sh

# Stop all servers
./stop-servers.sh
```

## Performance Optimization

### 1. TCP Parameter Optimization (Execute on all servers)

```bash
sudo sysctl -w net.core.rmem_max=268435456
sudo sysctl -w net.core.wmem_max=268435456
sudo sysctl -w net.ipv4.tcp_rmem="4096 87380 268435456"
sudo sysctl -w net.ipv4.tcp_wmem="4096 65536 268435456"
```

### 2. Process Number Optimization

Adjust based on CPU core count:
```bash
python server.py --server-id 1 --vdpf-processes 32  # For 64-core machines
```

## Dataset Support

| Dataset | Neighbors (K) | Nodes | Status |
|---------|-----------|--------|--------|
| siftsmall | 100 | 10,000 | ✅ Fully Supported |
| laion | 36 | 100,000 | ✅ Fully Supported |
| nfcorpus | 10 | 3,633 | ✅ Fully Supported |
| tripclick | 36 | 1.5M | 🟡 Requires Data Preparation |

## Monitoring Features

The system automatically records and displays:
- Execution time for each phase
- Network transfer size and speed
- Neighbor list query accuracy
- Detailed performance statistics

## Test Reports

Test results are automatically saved to `nl_result.md`, containing:
- Detailed query time breakdown
- Network transfer statistics
- Accuracy assessment
- Performance analysis

## Troubleshooting

### Connection Failed
- Check if security group rules allow ports 9001-9003
- Confirm servers are listening on correct ports
- Verify network connectivity

### Incomplete Data Error
- System has been optimized with MSG_WAITALL, should not occur
- If problem persists, check network stability

### Accuracy Issues
- Ensure neighbors_shares.npy data is generated correctly
- Verify groundtruth data format matches

## Advanced Features

### Custom Server Configuration

Create `custom_servers.json`:
```json
{
  "1": {"host": "192.168.1.101", "port": 9001},
  "2": {"host": "192.168.1.102", "port": 9002},
  "3": {"host": "192.168.1.103", "port": 9003}
}
```

Use custom configuration:
```bash
python client.py --config custom_servers.json
```

## Integration with Vector Query System

Complete vector search workflow:
1. Use `distributed-nl` to query K nearest neighbor indices
2. Use `distributed-deploy` to retrieve vector content of these neighbors
3. Calculate exact similarity and sort on client side

This separation design provides better flexibility and performance optimization opportunities.