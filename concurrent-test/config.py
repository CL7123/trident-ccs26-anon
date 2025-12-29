#!/usr/bin/env python3
"""
并发测试配置文件
"""

# 服务器配置（客户端使用公网IP）
SERVERS = {
    1: {"host": "192.168.1.101", "port": 8001},
    2: {"host": "192.168.1.102", "port": 8002},
    3: {"host": "192.168.1.103", "port": 8003}
}

# 默认数据集
DEFAULT_DATASET = "siftsmall"

# 并发测试配置
CONCURRENT_LEVELS = [1, 2, 4, 8, 16, 32, 64]  # 测试的并发级别
QUERIES_PER_LEVEL = 100  # 每个并发级别的查询数量
WARMUP_QUERIES = 10  # 预热查询数量
RECOVERY_WAIT_TIME = 5  # 每个测试级别后的恢复等待时间（秒）
