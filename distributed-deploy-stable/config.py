#!/usr/bin/env python3
"""
分布式部署配置文件
请根据实际的服务器IP地址修改此配置
"""

# 服务器配置
# 格式: server_id -> {"host": "IP地址", "port": 端口号}
SERVERS = {
    1: {
        "host": "192.168.1.101",  # 第一台服务器的公网IP
        "port": 8001
    },
    2: {
        "host": "192.168.1.102",  # 第二台服务器的公网IP
        "port": 8002
    },
    3: {
        "host": "192.168.1.103",  # 第三台服务器的公网IP
        "port": 8003
    }
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
请确保所有服务器的安全组/防火墙规则允许以下端口的入站流量：
- 8001 (Server 1)
- 8002 (Server 2)
- 8003 (Server 3)

对于AWS EC2实例：
1. 进入EC2控制台
2. 选择实例 -> 安全 -> 安全组
3. 编辑入站规则，添加：
   - 类型: 自定义TCP
   - 端口范围: 8001-8003
   - 来源: 0.0.0.0/0 (或限制为特定IP范围)
"""