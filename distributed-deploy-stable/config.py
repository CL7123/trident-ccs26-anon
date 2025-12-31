#!/usr/bin/env python3
"""
[CN]
[CN]IP[CN]
"""

# [CN]
# [CN]: server_id -> {"host": "IP[CN]", "port": [CN]}
SERVERS = {
    1: {
        "host": "192.168.1.101",  # [CN]IP
        "port": 8001
    },
    2: {
        "host": "192.168.1.102",  # [CN]IP
        "port": 8002
    },
    3: {
        "host": "192.168.1.103",  # [CN]IP
        "port": 8003
    }
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
[CN]/[CN]：
- 8001 (Server 1)
- 8002 (Server 2)
- 8003 (Server 3)

[CN]AWS EC2instance：
1. [CN]EC2[CN]
2. [CN]instance -> [CN] -> [CN]
3. [CN]，[CN]：
   - [CN]: [CN]TCP
   - [CN]: 8001-8003
   - [CN]: 0.0.0.0/0 ([CN]IP[CN])
"""