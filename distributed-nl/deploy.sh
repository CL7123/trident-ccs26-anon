#!/bin/bash

# [CN]deploy[CN]

echo "=== deploy[CN] ==="

# [CN]serverIP
SERVER1="192.168.1.101"
SERVER2="192.168.1.102"
SERVER3="192.168.1.103"

# SSH[CN]path
SSH_KEY="~/.ssh/your-key.pem"

# [CN]file
if [ ! -f "$SSH_KEY" ]; then
    echo "[CN]: SSH[CN]file[CN]: $SSH_KEY"
    exit 1
fi

echo "[CN]file[CN]server..."

# [CN]server1
echo "[CN]server1 ($SERVER1)..."
rsync -avz -e "ssh -i $SSH_KEY" \
    --exclude='__pycache__' \
    --exclude='*.pyc' \
    --exclude='nl_result.md' \
    ~/trident/distributed-nl/ \
    ubuntu@$SERVER1:~/trident/distributed-nl/

# [CN]server2
echo "[CN]server2 ($SERVER2)..."
rsync -avz -e "ssh -i $SSH_KEY" \
    --exclude='__pycache__' \
    --exclude='*.pyc' \
    --exclude='nl_result.md' \
    ~/trident/distributed-nl/ \
    ubuntu@$SERVER2:~/trident/distributed-nl/

# [CN]server3
echo "[CN]server3 ($SERVER3)..."
rsync -avz -e "ssh -i $SSH_KEY" \
    --exclude='__pycache__' \
    --exclude='*.pyc' \
    --exclude='nl_result.md' \
    ~/trident/distributed-nl/ \
    ubuntu@$SERVER3:~/trident/distributed-nl/

echo "deploy[CN]！"
echo ""
echo "[CN]："
echo "1. [CN]server[CN]start[CN]："
echo "   ssh -i $SSH_KEY ubuntu@<server-ip>"
echo "   cd ~/trident/distributed-nl"
echo "   python server.py --server-id <1/2/3> --dataset siftsmall"
echo ""
echo "2. [CN]clientruntest："
echo "   python client.py --dataset siftsmall --num-queries 10"