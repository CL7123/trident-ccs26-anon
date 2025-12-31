#!/usr/bin/env python3
"""
[CN]
"""

# [CN]（Client uses public IP）
SERVERS = {
    1: {"host": "192.168.1.101", "port": 8001},
    2: {"host": "192.168.1.102", "port": 8002},
    3: {"host": "192.168.1.103", "port": 8003}
}

# [CN]Dataset
DEFAULT_DATASET = "siftsmall"

# [CN]
CONCURRENT_LEVELS = [1, 2, 4, 8, 16, 32, 64]  # [CN]
QUERIES_PER_LEVEL = 100  # [CN]Number of queries[CN]
WARMUP_QUERIES = 10  # [CN]Number of queries[CN]
RECOVERY_WAIT_TIME = 5  # [CN]（[CN]）
