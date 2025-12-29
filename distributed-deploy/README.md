# 分布式部署指南

本目录包含了在真实网络环境中部署 Trident 系统的所有必要文件。

## 目录结构

```
distributed-deploy/
├── server.py          # 分布式服务器代码
├── client.py          # 分布式客户端代码
├── config.py          # 网络配置文件
├── deploy.sh          # 自动化部署脚本
└── README.md          # 本文档
```

## 前置要求

1. **三台服务器**：需要3台可以互相通信的服务器（推荐使用AWS EC2实例）
2. **Python环境**：每台服务器上需要Python 3.8+
3. **SSH访问**：能够通过SSH访问所有服务器
4. **防火墙配置**：开放端口 8001-8003

## 快速开始

### 1. 配置服务器信息

编辑 `config.py` 文件，填入实际的服务器IP地址：

```python
SERVERS = {
    1: {"host": "192.168.1.101", "port": 8001},
    2: {"host": "YOUR_SERVER_2_IP", "port": 8002},
    3: {"host": "YOUR_SERVER_3_IP", "port": 8003}
}
```

### 2. 配置部署脚本

编辑 `deploy.sh` 文件，更新服务器IP：

```bash
SERVERS[1]="192.168.1.101"
SERVERS[2]="YOUR_SERVER_2_IP"
SERVERS[3]="YOUR_SERVER_3_IP"
```

### 3. 部署服务器

使用部署脚本一键部署：

```bash
./deploy.sh
```

选择选项 1 来部署并启动所有服务器。

### 4. 运行客户端测试

在任意一台机器上运行客户端：

```bash
cd ~/trident/distributed-deploy
python3 client.py --dataset siftsmall --num-queries 10
```

## 手动部署步骤

如果需要手动部署，请按以下步骤操作：

### 在每台服务器上：

1. **同步代码**：
```bash
rsync -avz -e "ssh -i your-key.pem" --exclude='venv/' ./ ubuntu@SERVER_IP:~/test/
```

2. **启动服务器**（在各自的服务器上）：
```bash
# 服务器1
python3 server.py --server-id 1 --dataset siftsmall

# 服务器2
python3 server.py --server-id 2 --dataset siftsmall

# 服务器3
python3 server.py --server-id 3 --dataset siftsmall
```

### 在客户端机器上：

```bash
python3 client.py --dataset siftsmall --num-queries 10
```

## 命令行参数

### 服务器参数

- `--server-id`: 服务器ID (1, 2, 或 3)
- `--dataset`: 数据集名称 (siftsmall, laion, tripclick, ms_marco, nfcorpus)
- `--vdpf-processes`: VDPF评估进程数 (默认: 4)

### 客户端参数

- `--dataset`: 数据集名称 (默认: siftsmall)
- `--num-queries`: 测试查询数量 (默认: 10)
- `--no-report`: 不保存测试报告
- `--config`: 自定义配置文件路径
- `--status-only`: 只获取服务器状态

## 网络要求

### AWS EC2 安全组配置

1. 登录AWS控制台
2. 进入EC2 -> 实例 -> 选择实例 -> 安全 -> 安全组
3. 编辑入站规则，添加：
   - 类型: 自定义TCP
   - 端口范围: 8001-8003
   - 来源: 0.0.0.0/0 (或限制为特定IP)

### 本地防火墙配置（如果需要）

```bash
# Ubuntu/Debian
sudo ufw allow 8001:8003/tcp

# CentOS/RHEL
sudo firewall-cmd --permanent --add-port=8001-8003/tcp
sudo firewall-cmd --reload
```

## 故障排除

### 1. 连接超时

- 检查防火墙/安全组配置
- 验证服务器IP地址是否正确
- 确保服务器正在运行

### 2. SSH连接失败

- 检查密钥文件权限：`chmod 400 your-key.pem`
- 验证用户名（ubuntu/ec2-user）
- 检查SSH端口是否开放

### 3. 服务启动失败

- 检查Python依赖是否安装完整
- 查看服务器日志：`tail -f server_X.log`
- 确保数据文件存在于正确位置

### 4. 查询失败

- 确保至少2个服务器正常运行
- 检查网络连接是否稳定
- 验证数据集是否正确加载

## 性能优化建议

1. **网络优化**：
   - 将服务器部署在同一地区（AWS Region）
   - 使用内网IP进行通信（如果可能）
   - 启用TCP_NODELAY减少延迟

2. **进程配置**：
   - 根据CPU核心数调整 `--vdpf-processes`
   - 监控CPU和内存使用情况

3. **数据优化**：
   - 使用SSD存储数据文件
   - 预热进程池以减少启动延迟

## 监控和日志

- 服务器日志位置：`~/test/distributed-deploy/server_X.log`
- 客户端测试报告：`~/test/distributed-deploy/distributed_result.md`

使用部署脚本查看日志：
```bash
./deploy.sh
# 选择选项 7
```

## 安全注意事项

1. **生产环境建议**：
   - 使用TLS加密通信
   - 限制IP访问范围
   - 定期更新系统和依赖

2. **密钥管理**：
   - 妥善保管SSH密钥
   - 不要将密钥提交到版本控制

## 联系支持

如遇到问题，请检查：
1. 服务器日志文件
2. 网络连接状态
3. 防火墙配置

详细的错误信息将有助于快速定位和解决问题。