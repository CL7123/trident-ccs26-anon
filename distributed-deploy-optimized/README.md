# Distributed Deployment Guide

This directory contains all necessary files for deploying the Trident system in a real network environment.

## Directory Structure

```
distributed-deploy/
├── server.py          # Distributed server code
├── client.py          # Distributed client code
├── config.py          # Network configuration file
├── deploy.sh          # Automated deployment script
└── README.md          # This document
```

## Prerequisites

1. **Three servers**: 3 servers that can communicate with each other (AWS EC2 instances recommended)
2. **Python environment**: Python 3.8+ on each server
3. **SSH access**: Ability to access all servers via SSH
4. **Firewall configuration**: Open ports 8001-8003

## Quick Start

### 1. Configure Server Information

Edit the `config.py` file and enter the actual server IP addresses:

```python
SERVERS = {
    1: {"host": "192.168.1.101", "port": 8001},
    2: {"host": "YOUR_SERVER_2_IP", "port": 8002},
    3: {"host": "YOUR_SERVER_3_IP", "port": 8003}
}
```

### 2. Configure Deployment Script

Edit the `deploy.sh` file and update the server IPs:

```bash
SERVERS[1]="192.168.1.101"
SERVERS[2]="YOUR_SERVER_2_IP"
SERVERS[3]="YOUR_SERVER_3_IP"
```

### 3. Deploy Servers

Use the deployment script for one-click deployment:

```bash
./deploy.sh
```

Select option 1 to deploy and start all servers.

### 4. Run Client Tests

Run the client on any machine:

```bash
cd ~/trident/distributed-deploy
python3 client.py --dataset siftsmall --num-queries 10
```

## Manual Deployment Steps

If you need to deploy manually, follow these steps:

### On Each Server:

1. **Synchronize code**:
```bash
rsync -avz -e "ssh -i your-key.pem" --exclude='venv/' ./ ubuntu@SERVER_IP:~/test/
```

2. **Start servers** (on their respective servers):
```bash
# Server 1
python3 server.py --server-id 1 --dataset siftsmall

# Server 2
python3 server.py --server-id 2 --dataset siftsmall

# Server 3
python3 server.py --server-id 3 --dataset siftsmall
```

### On Client Machine:

```bash
python3 client.py --dataset siftsmall --num-queries 10
```

## Command Line Parameters

### Server Parameters

- `--server-id`: Server ID (1, 2, or 3)
- `--dataset`: Dataset name (siftsmall, laion, tripclick, ms_marco, nfcorpus)
- `--vdpf-processes`: Number of VDPF evaluation processes (default: 4)

### Client Parameters

- `--dataset`: Dataset name (default: siftsmall)
- `--num-queries`: Number of test queries (default: 10)
- `--no-report`: Do not save test reports
- `--config`: Custom configuration file path
- `--status-only`: Only get server status

## Network Requirements

### AWS EC2 Security Group Configuration

1. Log in to AWS Console
2. Go to EC2 -> Instances -> Select Instance -> Security -> Security Groups
3. Edit inbound rules, add:
   - Type: Custom TCP
   - Port range: 8001-8003
   - Source: 0.0.0.0/0 (or restrict to specific IP)

### Local Firewall Configuration (if needed)

```bash
# Ubuntu/Debian
sudo ufw allow 8001:8003/tcp

# CentOS/RHEL
sudo firewall-cmd --permanent --add-port=8001-8003/tcp
sudo firewall-cmd --reload
```

## Troubleshooting

### 1. Connection Timeout

- Check firewall/security group configuration
- Verify server IP addresses are correct
- Ensure servers are running

### 2. SSH Connection Failed

- Check key file permissions: `chmod 400 your-key.pem`
- Verify username (ubuntu/ec2-user)
- Check if SSH port is open

### 3. Service Startup Failed

- Check if Python dependencies are installed completely
- View server logs: `tail -f server_X.log`
- Ensure data files exist at correct location

### 4. Query Failed

- Ensure at least 2 servers are running properly
- Check network connection stability
- Verify dataset is loaded correctly

## Performance Optimization Tips

1. **Network optimization**:
   - Deploy servers in the same AWS region
   - Use private IPs for communication (if possible)
   - Enable TCP_NODELAY to reduce latency

2. **Process configuration**:
   - Adjust `--vdpf-processes` based on CPU core count
   - Monitor CPU and memory usage

3. **Data optimization**:
   - Use SSD for storing data files
   - Warm up process pool to reduce startup delay

## Monitoring and Logs

- Server log location: `~/test/distributed-deploy/server_X.log`
- Client test report: `~/test/distributed-deploy/distributed_result.md`

View logs using deployment script:
```bash
./deploy.sh
# Select option 7
```

## Security Considerations

1. **Production environment recommendations**:
   - Use TLS encryption for communication
   - Restrict IP access range
   - Keep systems and dependencies up to date

2. **Key management**:
   - Keep SSH keys secure
   - Do not commit keys to version control

## Support

If you encounter issues, check:
1. Server log files
2. Network connection status
3. Firewall configuration

Detailed error messages will help quickly identify and resolve issues.