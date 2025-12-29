#!/usr/bin/env python3
"""
分布式部署配置文件
请根据实际的服务器IP地址修改此配置
"""

# 服务器配置
# 客户端使用公网IP连接服务器，服务器间使用私网IP通信
SERVERS = {
    1: {
        "public_host": "192.168.1.101",   # 客户端连接用的公网IP
        "private_host": "10.0.1.101",   # 服务器间通信用的私网IP
        "port": 8001
    },
    2: {
        "public_host": "192.168.1.102",    # 客户端连接用的公网IP
        "private_host": "10.0.1.102",   # 服务器间通信用的私网IP
        "port": 8002
    },
    3: {
        "public_host": "192.168.1.101",  # 客户端连接用的公网IP
        "private_host": "10.0.1.103",    # 服务器间通信用的私网IP
        "port": 8003
    }
}

# 为了向后兼容，客户端默认使用公网IP
CLIENT_SERVERS = {
    server_id: {
        "host": config["public_host"],
        "port": config["port"]
    }
    for server_id, config in SERVERS.items()
}

# 服务器间通信使用私网IP
SERVER_TO_SERVER = {
    server_id: {
        "host": config["private_host"],
        "port": config["port"]
    }
    for server_id, config in SERVERS.items()
}

# 数据集配置
DEFAULT_DATASET = "siftsmall"

# 网络配置
CONNECTION_TIMEOUT = 30  # 连接超时时间（秒）
RETRY_COUNT = 3         # 重试次数
RETRY_DELAY = 2         # 重试延迟（秒）

# 进程配置
DEFAULT_VDPF_PROCESSES = 4  # 默认VDPF评估进程数

# 日志配置
LOG_LEVEL = "INFO"

# 交换目录（所有服务器必须使用相同的路径）
EXCHANGE_DIR = "/tmp/mpc_exchange"

# 安全组/防火墙规则提醒
"""
网络通信配置：
- 客户端 → 服务器：使用公网IP（需要公网访问权限）
- 服务器 ↔ 服务器：使用私网IP（VPC内部通信）

请确保所有服务器的安全组/防火墙规则配置：

1. 客户端访问（公网入站）：
   - 类型: 自定义TCP
   - 端口范围: 8001-8003
   - 来源: 0.0.0.0/0 (或限制为客户端IP范围)

2. 服务器间通信（私网入站）：
   - 类型: 自定义TCP
   - 端口范围: 8001-8003
   - 来源: VPC CIDR (如 10.0.0.0/16) 或安全组自引用

优势：
- 客户端可通过公网访问服务器
- 服务器间通信使用私网，更安全、延迟更低
- 节省公网带宽费用
"""