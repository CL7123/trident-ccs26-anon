#!/usr/bin/env python3
"""
Concurrent test configuration file
"""

# Server configuration (client uses public IPs)
SERVERS = {
    1: {"host": "192.168.1.101", "port": 8001},
    2: {"host": "192.168.1.102", "port": 8002},
    3: {"host": "192.168.1.103", "port": 8003}
}

# Default dataset
DEFAULT_DATASET = "siftsmall"

# Concurrent test configuration
CONCURRENT_LEVELS = [1, 2, 4, 8, 16, 32, 64]  # Concurrency levels to test
QUERIES_PER_LEVEL = 100  # Number of queries per concurrency level
WARMUP_QUERIES = 10  # Number of warmup queries
RECOVERY_WAIT_TIME = 5  # Recovery wait time after each test level (seconds)
