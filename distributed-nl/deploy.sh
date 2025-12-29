#!/bin/bash

# 分布式邻居列表系统部署脚本

echo "=== 部署分布式邻居列表系统 ==="

# 定义服务器IP
SERVER1="192.168.1.101"
SERVER2="192.168.1.102"
SERVER3="192.168.1.103"

# SSH密钥路径
SSH_KEY="~/.ssh/your-key.pem"

# 检查密钥文件
if [ ! -f "$SSH_KEY" ]; then
    echo "错误: SSH密钥文件不存在: $SSH_KEY"
    exit 1
fi

echo "同步文件到所有服务器..."

# 同步到服务器1
echo "同步到服务器1 ($SERVER1)..."
rsync -avz -e "ssh -i $SSH_KEY" \
    --exclude='__pycache__' \
    --exclude='*.pyc' \
    --exclude='nl_result.md' \
    ~/trident/distributed-nl/ \
    ubuntu@$SERVER1:~/trident/distributed-nl/

# 同步到服务器2
echo "同步到服务器2 ($SERVER2)..."
rsync -avz -e "ssh -i $SSH_KEY" \
    --exclude='__pycache__' \
    --exclude='*.pyc' \
    --exclude='nl_result.md' \
    ~/trident/distributed-nl/ \
    ubuntu@$SERVER2:~/trident/distributed-nl/

# 同步到服务器3
echo "同步到服务器3 ($SERVER3)..."
rsync -avz -e "ssh -i $SSH_KEY" \
    --exclude='__pycache__' \
    --exclude='*.pyc' \
    --exclude='nl_result.md' \
    ~/trident/distributed-nl/ \
    ubuntu@$SERVER3:~/trident/distributed-nl/

echo "部署完成！"
echo ""
echo "下一步："
echo "1. 在每台服务器上启动邻居列表服务："
echo "   ssh -i $SSH_KEY ubuntu@<server-ip>"
echo "   cd ~/trident/distributed-nl"
echo "   python server.py --server-id <1/2/3> --dataset siftsmall"
echo ""
echo "2. 在客户端运行测试："
echo "   python client.py --dataset siftsmall --num-queries 10"