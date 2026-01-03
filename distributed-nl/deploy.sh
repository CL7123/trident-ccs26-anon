#!/bin/bash

# Distributed neighbor list system deployment script

echo "=== Deploy Distributed Neighbor List System ==="

# Define server IPs
SERVER1="192.168.1.101"
SERVER2="192.168.1.102"
SERVER3="192.168.1.103"

# SSH key path
SSH_KEY="~/.ssh/your-key.pem"

# Check SSH key file
if [ ! -f "$SSH_KEY" ]; then
    echo "Error: SSH key file not found: $SSH_KEY"
    exit 1
fi

echo "Syncing files to all servers..."

# Sync to Server 1
echo "Syncing to Server 1 ($SERVER1)..."
rsync -avz -e "ssh -i $SSH_KEY" \
    --exclude='__pycache__' \
    --exclude='*.pyc' \
    --exclude='nl_result.md' \
    ~/trident/distributed-nl/ \
    ubuntu@$SERVER1:~/trident/distributed-nl/

# Sync to Server 2
echo "Syncing to Server 2 ($SERVER2)..."
rsync -avz -e "ssh -i $SSH_KEY" \
    --exclude='__pycache__' \
    --exclude='*.pyc' \
    --exclude='nl_result.md' \
    ~/trident/distributed-nl/ \
    ubuntu@$SERVER2:~/trident/distributed-nl/

# Sync to Server 3
echo "Syncing to Server 3 ($SERVER3)..."
rsync -avz -e "ssh -i $SSH_KEY" \
    --exclude='__pycache__' \
    --exclude='*.pyc' \
    --exclude='nl_result.md' \
    ~/trident/distributed-nl/ \
    ubuntu@$SERVER3:~/trident/distributed-nl/

echo "Deployment complete!"
echo ""
echo "Next steps:"
echo "1. Start the neighbor list service on each server:"
echo "   ssh -i $SSH_KEY ubuntu@<server-ip>"
echo "   cd ~/trident/distributed-nl"
echo "   python server.py --server-id <1/2/3> --dataset siftsmall"
echo ""
echo "2. Run tests on the client:"
echo "   python client.py --dataset siftsmall --num-queries 10"