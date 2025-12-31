#!/usr/bin/env python3
"""
[CN]
[CN]IP[CN]
"""

# [CN]
# Client uses public IPconnect[CN]，[CN]IP[CN]
SERVERS = {
    1: {
        "public_host": "192.168.1.101",   # [CN]connect[CN]IP
        "private_host": "10.0.1.101",   # [CN]IP（[CN]server1[CN]）
        "port": 8001
    },
    2: {
        "public_host": "192.168.1.102",    # [CN]connect[CN]IP
        "private_host": "10.0.1.102",   # [CN]IP（[CN]server2[CN]）
        "port": 8002
    },
    3: {
        "public_host": "192.168.1.103",  # [CN]connect[CN]IP
        "private_host": "10.0.1.103",    # [CN]IP（[CN]server3[CN]）
        "port": 8003
    }
}

# [CN]，[CN]IP
CLIENT_SERVERS = {
    server_id: {
        "host": config["public_host"],
        "port": config["port"]
    }
    for server_id, config in SERVERS.items()
}

# [CN]IP
SERVER_TO_SERVER = {
    server_id: {
        "host": config["private_host"],
        "port": config["port"]
    }
    for server_id, config in SERVERS.items()
}

# Dataset[CN]
DEFAULT_DATASET = "siftsmall"

# [CN]
CONNECTION_TIMEOUT = 30  # connect[CN]（[CN]）
RETRY_COUNT = 3         # [CN]
RETRY_DELAY = 2         # [CN]（[CN]）

# [CN]
DEFAULT_VDPF_PROCESSES = 4  # [CN]VDPF[CN]

# [CN]
LOG_LEVEL = "INFO"

# [CN]（[CN]）
EXCHANGE_DIR = "/tmp/mpc_exchange"

# [CN]/[CN]
"""
[CN]：
- [CN] → [CN]：[CN]IP（[CN]）
- [CN] ↔ [CN]：[CN]IP（VPC[CN]）

[CN]/[CN]：

1. [CN]（[CN]）：
   - [CN]: [CN]TCP
   - [CN]: 8001-8003
   - [CN]: 0.0.0.0/0 ([CN]IP[CN])

2. [CN]（[CN]）：
   - [CN]: [CN]TCP
   - [CN]: 8001-8003
   - [CN]: VPC CIDR ([CN] 10.0.0.0/16) [CN]

[CN]：
- [CN]
- [CN]，[CN]、[CN]
- [CN]
"""