#!/bin/bash

# Distributed deployment script
# Used to deploy code to multiple servers and start services

set -e

# Color definitions
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
PROJECT_DIR="~/trident"
REMOTE_USER="ubuntu"
PEM_FILE="~/.ssh/your-key.pem"

# Server list (modify based on actual situation)
declare -A SERVERS
SERVERS[1]="192.168.1.101"
SERVERS[2]="192.168.1.102"
SERVERS[3]="192.168.1.102"

# Function: Print colored message
print_message() {
    local color=$1
    local message=$2
    echo -e "${color}${message}${NC}"
}

# Function: Check SSH connection
check_ssh_connection() {
    local server_ip=$1
    local server_id=$2

    print_message $YELLOW "Checking SSH connection to server $server_id ($server_ip)..."

    if ssh -i "$PEM_FILE" -o ConnectTimeout=10 -o StrictHostKeyChecking=no \
        ${REMOTE_USER}@${server_ip} "echo 'SSH connection successful'" &> /dev/null; then
        print_message $GREEN "✓ Server $server_id SSH connection normal"
        return 0
    else
        print_message $RED "✗ Unable to connect to server $server_id"
        return 1
    fi
}

# Function: Sync code to remote server
sync_code() {
    local server_ip=$1
    local server_id=$2

    print_message $YELLOW "Syncing code to server $server_id ($server_ip)..."

    # Use rsync to sync code, exclude virtual environment
    rsync -avz -e "ssh -i $PEM_FILE -o StrictHostKeyChecking=no" \
        --exclude='venv/' \
        --exclude='*.pyc' \
        --exclude='__pycache__/' \
        --exclude='.git/' \
        "$PROJECT_DIR/" \
        ${REMOTE_USER}@${server_ip}:~/test/

    if [ $? -eq 0 ]; then
        print_message $GREEN "✓ Code synced to server $server_id successfully"
    else
        print_message $RED "✗ Code sync to server $server_id failed"
        return 1
    fi
}

# Function: Start remote server
start_remote_server() {
    local server_ip=$1
    local server_id=$2
    local dataset=$3

    print_message $YELLOW "Starting service on server $server_id..."

    # SSH to remote server and start service
    ssh -i "$PEM_FILE" -o StrictHostKeyChecking=no ${REMOTE_USER}@${server_ip} << EOF
        cd ~/test/distributed-deploy

        # Check if service is already running
        if pgrep -f "server.py --server-id $server_id" > /dev/null; then
            echo "Stopping existing service..."
            pkill -f "server.py --server-id $server_id"
            sleep 2
        fi

        # Start new service
        echo "Starting server $server_id..."
        nohup python3 server.py --server-id $server_id --dataset $dataset > server_${server_id}.log 2>&1 &

        # Wait for service to start
        sleep 3

        # Check if service started successfully
        if pgrep -f "server.py --server-id $server_id" > /dev/null; then
            echo "✓ Server $server_id started successfully"
        else
            echo "✗ Server $server_id failed to start"
            tail -n 20 server_${server_id}.log
            exit 1
        fi
EOF

    if [ $? -eq 0 ]; then
        print_message $GREEN "✓ Server $server_id started successfully"
    else
        print_message $RED "✗ Server $server_id failed to start"
        return 1
    fi
}

# Function: Stop remote server
stop_remote_server() {
    local server_ip=$1
    local server_id=$2

    print_message $YELLOW "Stopping service on server $server_id..."

    ssh -i "$PEM_FILE" -o StrictHostKeyChecking=no ${REMOTE_USER}@${server_ip} << EOF
        if pgrep -f "server.py --server-id $server_id" > /dev/null; then
            pkill -f "server.py --server-id $server_id"
            echo "✓ Service stopped"
        else
            echo "Service not running"
        fi
EOF
}

# Function: Check remote server status
check_server_status() {
    local server_ip=$1
    local server_id=$2

    print_message $YELLOW "Checking server $server_id status..."

    ssh -i "$PEM_FILE" -o StrictHostKeyChecking=no ${REMOTE_USER}@${server_ip} << EOF
        if pgrep -f "server.py --server-id $server_id" > /dev/null; then
            echo "✓ Service is running"
            echo "Latest logs:"
            tail -n 10 ~/test/distributed-deploy/server_${server_id}.log
        else
            echo "✗ Service not running"
        fi
EOF
}

# Main menu
show_menu() {
    echo
    print_message $GREEN "=== Distributed Deployment Management Tool ==="
    echo "1. Deploy and start all servers"
    echo "2. Sync code only"
    echo "3. Start all servers"
    echo "4. Stop all servers"
    echo "5. Check server status"
    echo "6. Deploy single server"
    echo "7. View server logs"
    echo "8. Exit"
    echo
}

# Function: Deploy all servers
deploy_all() {
    local dataset=${1:-siftsmall}

    print_message $GREEN "Starting deployment of all servers (dataset: $dataset)"

    local success=true

    for server_id in "${!SERVERS[@]}"; do
        server_ip="${SERVERS[$server_id]}"

        if [[ "$server_ip" == "YOUR_SERVER_"* ]]; then
            print_message $RED "Please configure IP address for server $server_id in the script first"
            continue
        fi

        # Check SSH connection
        if ! check_ssh_connection "$server_ip" "$server_id"; then
            success=false
            continue
        fi

        # Sync code
        if ! sync_code "$server_ip" "$server_id"; then
            success=false
            continue
        fi

        # Start server
        if ! start_remote_server "$server_ip" "$server_id" "$dataset"; then
            success=false
            continue
        fi
    done

    if $success; then
        print_message $GREEN "All servers deployed successfully!"
    else
        print_message $RED "Some servers failed to deploy"
    fi
}

# Main program
main() {
    # Check if PEM file exists
    if [ ! -f "$PEM_FILE" ]; then
        print_message $RED "Error: Key file $PEM_FILE not found"
        exit 1
    fi

    # Set key file permissions
    chmod 400 "$PEM_FILE"

    while true; do
        show_menu
        read -p "Select operation (1-8): " choice

        case $choice in
            1)
                read -p "Enter dataset name (default: siftsmall): " dataset
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
                read -p "Enter dataset name (default: siftsmall): " dataset
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
                read -p "Enter server ID (1-3): " server_id
                if [[ -n "${SERVERS[$server_id]}" ]]; then
                    server_ip="${SERVERS[$server_id]}"
                    if [[ "$server_ip" != "YOUR_SERVER_"* ]]; then
                        read -p "Enter dataset name (default: siftsmall): " dataset
                        dataset=${dataset:-siftsmall}
                        check_ssh_connection "$server_ip" "$server_id" && \
                        sync_code "$server_ip" "$server_id" && \
                        start_remote_server "$server_ip" "$server_id" "$dataset"
                    else
                        print_message $RED "Please configure IP address for server $server_id first"
                    fi
                else
                    print_message $RED "Invalid server ID"
                fi
                ;;
            7)
                read -p "Enter server ID (1-3): " server_id
                if [[ -n "${SERVERS[$server_id]}" ]]; then
                    server_ip="${SERVERS[$server_id]}"
                    if [[ "$server_ip" != "YOUR_SERVER_"* ]]; then
                        print_message $YELLOW "Latest logs for server $server_id:"
                        ssh -i "$PEM_FILE" -o StrictHostKeyChecking=no \
                            ${REMOTE_USER}@${server_ip} \
                            "tail -n 50 ~/test/distributed-deploy/server_${server_id}.log"
                    else
                        print_message $RED "Please configure IP address for server $server_id first"
                    fi
                else
                    print_message $RED "Invalid server ID"
                fi
                ;;
            8)
                print_message $GREEN "Exiting deployment tool"
                exit 0
                ;;
            *)
                print_message $RED "Invalid selection"
                ;;
        esac
    done
}

# Run main program
main