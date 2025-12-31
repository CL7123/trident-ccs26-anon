#!/bin/bash

# [CN]deploy[CN]
# [CN]deploy[CN]server[CN]start[CN]

set -e

# [CN]
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# configure
PROJECT_DIR="~/trident"
REMOTE_USER="ubuntu"
PEM_FILE="~/.ssh/your-key.pem"

# server[CN]（[CN]）
declare -A SERVERS
SERVERS[1]="192.168.1.101"
SERVERS[2]="192.168.1.102"
SERVERS[3]="192.168.1.103"

# [CN]：[CN]
print_message() {
    local color=$1
    local message=$2
    echo -e "${color}${message}${NC}"
}

# [CN]：[CN]SSH[CN]
check_ssh_connection() {
    local server_ip=$1
    local server_id=$2
    
    print_message $YELLOW "[CN]server $server_id ($server_ip) [CN]SSH[CN]..."
    
    if ssh -i "$PEM_FILE" -o ConnectTimeout=10 -o StrictHostKeyChecking=no \
        ${REMOTE_USER}@${server_ip} "echo 'SSH[CN]'" &> /dev/null; then
        print_message $GREEN "✓ server $server_id SSH[CN]"
        return 0
    else
        print_message $RED "✗ [CN]server $server_id"
        return 1
    fi
}

# [CN]：[CN]server
sync_code() {
    local server_ip=$1
    local server_id=$2
    
    print_message $YELLOW "[CN]server $server_id ($server_ip)..."
    
    # [CN]rsync[CN]，[CN]
    rsync -avz -e "ssh -i $PEM_FILE -o StrictHostKeyChecking=no" \
        --exclude='venv/' \
        --exclude='*.pyc' \
        --exclude='__pycache__/' \
        --exclude='.git/' \
        "$PROJECT_DIR/" \
        ${REMOTE_USER}@${server_ip}:~/test/
    
    if [ $? -eq 0 ]; then
        print_message $GREEN "✓ [CN]server $server_id [CN]"
    else
        print_message $RED "✗ [CN]server $server_id [CN]"
        return 1
    fi
}

# [CN]：start[CN]server
start_remote_server() {
    local server_ip=$1
    local server_id=$2
    local dataset=$3
    
    print_message $YELLOW "[CN]server $server_id [CN]start[CN]..."
    
    # SSH[CN]server[CN]start[CN]
    ssh -i "$PEM_FILE" -o StrictHostKeyChecking=no ${REMOTE_USER}@${server_ip} << EOF
        cd ~/test/distributed-deploy
        
        # [CN]run
        if pgrep -f "server.py --server-id $server_id" > /dev/null; then
            echo "stop[CN]..."
            pkill -f "server.py --server-id $server_id"
            sleep 2
        fi
        
        # start[CN]
        echo "startserver $server_id..."
        nohup python3 server.py --server-id $server_id --dataset $dataset > server_${server_id}.log 2>&1 &
        
        # [CN]start
        sleep 3
        
        # [CN]start
        if pgrep -f "server.py --server-id $server_id" > /dev/null; then
            echo "✓ server $server_id start[CN]"
        else
            echo "✗ server $server_id start[CN]"
            tail -n 20 server_${server_id}.log
            exit 1
        fi
EOF
    
    if [ $? -eq 0 ]; then
        print_message $GREEN "✓ server $server_id start[CN]"
    else
        print_message $RED "✗ server $server_id start[CN]"
        return 1
    fi
}

# [CN]：stop[CN]server
stop_remote_server() {
    local server_ip=$1
    local server_id=$2
    
    print_message $YELLOW "stopserver $server_id [CN]..."
    
    ssh -i "$PEM_FILE" -o StrictHostKeyChecking=no ${REMOTE_USER}@${server_ip} << EOF
        if pgrep -f "server.py --server-id $server_id" > /dev/null; then
            pkill -f "server.py --server-id $server_id"
            echo "✓ [CN]stop"
        else
            echo "[CN]run"
        fi
EOF
}

# [CN]：[CN]server[CN]
check_server_status() {
    local server_ip=$1
    local server_id=$2
    
    print_message $YELLOW "[CN]server $server_id [CN]..."
    
    ssh -i "$PEM_FILE" -o StrictHostKeyChecking=no ${REMOTE_USER}@${server_ip} << EOF
        if pgrep -f "server.py --server-id $server_id" > /dev/null; then
            echo "✓ [CN]run"
            echo "[CN]："
            tail -n 10 ~/test/distributed-deploy/server_${server_id}.log
        else
            echo "✗ [CN]run"
        fi
EOF
}

# [CN]
show_menu() {
    echo
    print_message $GREEN "=== [CN]deploy[CN] ==="
    echo "1. deploy[CN]start[CN]server"
    echo "2. [CN]"
    echo "3. start[CN]server"
    echo "4. stop[CN]server"
    echo "5. [CN]server[CN]"
    echo "6. deploy[CN]server"
    echo "7. [CN]server[CN]"
    echo "8. [CN]"
    echo
}

# [CN]：deploy[CN]server
deploy_all() {
    local dataset=${1:-siftsmall}
    
    print_message $GREEN "[CN]deploy[CN]server（dataset: $dataset）"
    
    local success=true
    
    for server_id in "${!SERVERS[@]}"; do
        server_ip="${SERVERS[$server_id]}"
        
        if [[ "$server_ip" == "YOUR_SERVER_"* ]]; then
            print_message $RED "[CN]configureserver $server_id [CN]IP[CN]"
            continue
        fi
        
        # [CN]SSH[CN]
        if ! check_ssh_connection "$server_ip" "$server_id"; then
            success=false
            continue
        fi
        
        # [CN]
        if ! sync_code "$server_ip" "$server_id"; then
            success=false
            continue
        fi
        
        # startserver
        if ! start_remote_server "$server_ip" "$server_id" "$dataset"; then
            success=false
            continue
        fi
    done
    
    if $success; then
        print_message $GREEN "[CN]serverdeploy[CN]！"
    else
        print_message $RED "[CN]serverdeploy[CN]"
    fi
}

# [CN]
main() {
    # [CN]PEMfile[CN]
    if [ ! -f "$PEM_FILE" ]; then
        print_message $RED "[CN]：[CN]file $PEM_FILE"
        exit 1
    fi
    
    # [CN]file[CN]
    chmod 400 "$PEM_FILE"
    
    while true; do
        show_menu
        read -p "[CN] (1-8): " choice
        
        case $choice in
            1)
                read -p "[CN]inputdataset[CN] ([CN]: siftsmall): " dataset
                dataset=${dataset:-siftsmall}
                deploy_all "$dataset"
                ;;
            2)
                for server_id in "${!SERVERS[@]}"; do
                    server_ip="${SERVERS[$server_id]}"
                    if [[ "$server_ip" != "YOUR_SERVER_"* ]]; then
                        sync_code "$server_ip" "$server_id"
                    fi
                done
                ;;
            3)
                read -p "[CN]inputdataset[CN] ([CN]: siftsmall): " dataset
                dataset=${dataset:-siftsmall}
                for server_id in "${!SERVERS[@]}"; do
                    server_ip="${SERVERS[$server_id]}"
                    if [[ "$server_ip" != "YOUR_SERVER_"* ]]; then
                        start_remote_server "$server_ip" "$server_id" "$dataset"
                    fi
                done
                ;;
            4)
                for server_id in "${!SERVERS[@]}"; do
                    server_ip="${SERVERS[$server_id]}"
                    if [[ "$server_ip" != "YOUR_SERVER_"* ]]; then
                        stop_remote_server "$server_ip" "$server_id"
                    fi
                done
                ;;
            5)
                for server_id in "${!SERVERS[@]}"; do
                    server_ip="${SERVERS[$server_id]}"
                    if [[ "$server_ip" != "YOUR_SERVER_"* ]]; then
                        check_server_status "$server_ip" "$server_id"
                    fi
                done
                ;;
            6)
                read -p "[CN]inputserverID (1-3): " server_id
                if [[ -n "${SERVERS[$server_id]}" ]]; then
                    server_ip="${SERVERS[$server_id]}"
                    if [[ "$server_ip" != "YOUR_SERVER_"* ]]; then
                        read -p "[CN]inputdataset[CN] ([CN]: siftsmall): " dataset
                        dataset=${dataset:-siftsmall}
                        check_ssh_connection "$server_ip" "$server_id" && \
                        sync_code "$server_ip" "$server_id" && \
                        start_remote_server "$server_ip" "$server_id" "$dataset"
                    else
                        print_message $RED "[CN]configureserver $server_id [CN]IP[CN]"
                    fi
                else
                    print_message $RED "[CN]serverID"
                fi
                ;;
            7)
                read -p "[CN]inputserverID (1-3): " server_id
                if [[ -n "${SERVERS[$server_id]}" ]]; then
                    server_ip="${SERVERS[$server_id]}"
                    if [[ "$server_ip" != "YOUR_SERVER_"* ]]; then
                        print_message $YELLOW "server $server_id [CN]："
                        ssh -i "$PEM_FILE" -o StrictHostKeyChecking=no \
                            ${REMOTE_USER}@${server_ip} \
                            "tail -n 50 ~/test/distributed-deploy/server_${server_id}.log"
                    else
                        print_message $RED "[CN]configureserver $server_id [CN]IP[CN]"
                    fi
                else
                    print_message $RED "[CN]serverID"
                fi
                ;;
            8)
                print_message $GREEN "[CN]deploy[CN]"
                exit 0
                ;;
            *)
                print_message $RED "[CN]"
                ;;
        esac
    done
}

# run[CN]
main