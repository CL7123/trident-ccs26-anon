#!/usr/bin/env python3
"""
Distributed deployment configuration file
Please modify this configuration according to actual server IP addresses
"""

# Server configuration
# Format: server_id -> {"host": "IP address", "port": port number}
SERVERS = {
    1: {
        "host": "192.168.1.101",  # Public IP of first server
        "port": 8001
    },
    2: {
        "host": "192.168.1.102",  # Public IP of second server
        "port": 8002
    },
    3: {
        "host": "192.168.1.103",  # Public IP of third server
        "port": 8003
    }
}

# Dataset configuration
DEFAULT_DATASET = "siftsmall"

# Network configuration
CONNECTION_TIMEOUT = 30  # Connection timeout (seconds)
RETRY_COUNT = 3         # Number of retries
RETRY_DELAY = 2         # Retry delay (seconds)

# Process configuration
DEFAULT_VDPF_PROCESSES = 4  # Default number of VDPF evaluation processes

# Log configuration
LOG_LEVEL = "INFO"

# Exchange directory (all servers must use the same path)
EXCHANGE_DIR = "/tmp/mpc_exchange"

# Security group/Firewall rule reminder
"""
Please ensure all servers' security groups/firewall rules allow inbound traffic on the following ports:
- 8001 (Server 1)
- 8002 (Server 2)
- 8003 (Server 3)

For AWS EC2 instances:
1. Go to EC2 Console
2. Select Instance -> Security -> Security Groups
3. Edit inbound rules and add:
   - Type: Custom TCP
   - Port Range: 8001-8003
   - Source: 0.0.0.0/0 (or restrict to specific IP range)
"""