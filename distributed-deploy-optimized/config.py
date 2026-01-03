#!/usr/bin/env python3
"""
Distributed deployment configuration file
Please modify this configuration based on actual server IP addresses
"""

# Server configuration
# Clients use public IPs to connect to servers, servers use private IPs for inter-server communication
SERVERS = {
    1: {
        "public_host": "192.168.1.101",   # Public IP for client connections
        "private_host": "10.0.1.101",   # Private IP for inter-server communication
        "port": 8001
    },
    2: {
        "public_host": "192.168.1.102",    # Public IP for client connections
        "private_host": "10.0.1.102",   # Private IP for inter-server communication
        "port": 8002
    },
    3: {
        "public_host": "192.168.1.102",  # Public IP for client connections
        "private_host": "10.0.1.103",    # Private IP for inter-server communication
        "port": 8003
    }
}

# For backward compatibility, clients use public IPs by default
CLIENT_SERVERS = {
    server_id: {
        "host": config["public_host"],
        "port": config["port"]
    }
    for server_id, config in SERVERS.items()
}

# Inter-server communication uses private IPs
SERVER_TO_SERVER = {
    server_id: {
        "host": config["private_host"],
        "port": config["port"]
    }
    for server_id, config in SERVERS.items()
}

# Dataset configuration
DEFAULT_DATASET = "siftsmall"

# Network configuration
CONNECTION_TIMEOUT = 30  # Connection timeout in seconds
RETRY_COUNT = 3         # Number of retries
RETRY_DELAY = 2         # Retry delay in seconds

# Process configuration
DEFAULT_VDPF_PROCESSES = 4  # Default number of VDPF evaluation processes

# Log configuration
LOG_LEVEL = "INFO"

# Exchange directory (all servers must use the same path)
EXCHANGE_DIR = "/tmp/mpc_exchange"

# Security group/firewall rules reminder
"""
Network communication configuration:
- Client -> Server: Use public IP (requires public network access)
- Server <-> Server: Use private IP (VPC internal communication)

Ensure all servers have proper security group/firewall rules configured:

1. Client access (public inbound):
   - Type: Custom TCP
   - Port range: 8001-8003
   - Source: 0.0.0.0/0 (or restrict to client IP range)

2. Inter-server communication (private inbound):
   - Type: Custom TCP
   - Port range: 8001-8003
   - Source: VPC CIDR (e.g., 10.0.0.0/16) or security group self-reference

Benefits:
- Clients can access servers over public network
- Inter-server communication uses private network, more secure and lower latency
- Saves public bandwidth costs
"""