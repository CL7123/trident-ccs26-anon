#!/bin/bash

# 分布式部署脚本
# 用于将代码部署到多台服务器并启动服务

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 配置
PROJECT_DIR="~/trident"
REMOTE_USER="ubuntu"
PEM_FILE="~/.ssh/your-key.pem"

# 服务器列表（请根据实际情况修改）
declare -A SERVERS
SERVERS[1]="192.168.1.101"
SERVERS[2]="192.168.1.102"
SERVERS[3]="192.168.1.102"

# 函数：打印带颜色的消息
print_message() {
    local color=$1
    local message=$2
    echo -e "${color}${message}${NC}"
}

# 函数：检查SSH连接
check_ssh_connection() {
    local server_ip=$1
    local server_id=$2
    
    print_message $YELLOW "检查与服务器 $server_id ($server_ip) 的SSH连接..."
    
    if ssh -i "$PEM_FILE" -o ConnectTimeout=10 -o StrictHostKeyChecking=no \
        ${REMOTE_USER}@${server_ip} "echo 'SSH连接成功'" &> /dev/null; then
        print_message $GREEN "✓ 服务器 $server_id SSH连接正常"
        return 0
    else
        print_message $RED "✗ 无法连接到服务器 $server_id"
        return 1
    fi
}

# 函数：同步代码到远程服务器
sync_code() {
    local server_ip=$1
    local server_id=$2
    
    print_message $YELLOW "正在同步代码到服务器 $server_id ($server_ip)..."
    
    # 使用rsync同步代码，排除虚拟环境
    rsync -avz -e "ssh -i $PEM_FILE -o StrictHostKeyChecking=no" \
        --exclude='venv/' \
        --exclude='*.pyc' \
        --exclude='__pycache__/' \
        --exclude='.git/' \
        "$PROJECT_DIR/" \
        ${REMOTE_USER}@${server_ip}:~/test/
    
    if [ $? -eq 0 ]; then
        print_message $GREEN "✓ 代码同步到服务器 $server_id 成功"
    else
        print_message $RED "✗ 代码同步到服务器 $server_id 失败"
        return 1
    fi
}

# 函数：启动远程服务器
start_remote_server() {
    local server_ip=$1
    local server_id=$2
    local dataset=$3
    
    print_message $YELLOW "在服务器 $server_id 上启动服务..."
    
    # SSH到远程服务器并启动服务
    ssh -i "$PEM_FILE" -o StrictHostKeyChecking=no ${REMOTE_USER}@${server_ip} << EOF
        cd ~/test/distributed-deploy
        
        # 检查是否已有服务在运行
        if pgrep -f "server.py --server-id $server_id" > /dev/null; then
            echo "停止现有服务..."
            pkill -f "server.py --server-id $server_id"
            sleep 2
        fi
        
        # 启动新服务
        echo "启动服务器 $server_id..."
        nohup python3 server.py --server-id $server_id --dataset $dataset > server_${server_id}.log 2>&1 &
        
        # 等待服务启动
        sleep 3
        
        # 检查服务是否成功启动
        if pgrep -f "server.py --server-id $server_id" > /dev/null; then
            echo "✓ 服务器 $server_id 启动成功"
        else
            echo "✗ 服务器 $server_id 启动失败"
            tail -n 20 server_${server_id}.log
            exit 1
        fi
EOF
    
    if [ $? -eq 0 ]; then
        print_message $GREEN "✓ 服务器 $server_id 启动成功"
    else
        print_message $RED "✗ 服务器 $server_id 启动失败"
        return 1
    fi
}

# 函数：停止远程服务器
stop_remote_server() {
    local server_ip=$1
    local server_id=$2
    
    print_message $YELLOW "停止服务器 $server_id 上的服务..."
    
    ssh -i "$PEM_FILE" -o StrictHostKeyChecking=no ${REMOTE_USER}@${server_ip} << EOF
        if pgrep -f "server.py --server-id $server_id" > /dev/null; then
            pkill -f "server.py --server-id $server_id"
            echo "✓ 服务已停止"
        else
            echo "服务未在运行"
        fi
EOF
}

# 函数：检查远程服务器状态
check_server_status() {
    local server_ip=$1
    local server_id=$2
    
    print_message $YELLOW "检查服务器 $server_id 状态..."
    
    ssh -i "$PEM_FILE" -o StrictHostKeyChecking=no ${REMOTE_USER}@${server_ip} << EOF
        if pgrep -f "server.py --server-id $server_id" > /dev/null; then
            echo "✓ 服务正在运行"
            echo "最新日志："
            tail -n 10 ~/test/distributed-deploy/server_${server_id}.log
        else
            echo "✗ 服务未运行"
        fi
EOF
}

# 主菜单
show_menu() {
    echo
    print_message $GREEN "=== 分布式部署管理工具 ==="
    echo "1. 部署并启动所有服务器"
    echo "2. 仅同步代码"
    echo "3. 启动所有服务器"
    echo "4. 停止所有服务器"
    echo "5. 检查服务器状态"
    echo "6. 部署单个服务器"
    echo "7. 查看服务器日志"
    echo "8. 退出"
    echo
}

# 函数：部署所有服务器
deploy_all() {
    local dataset=${1:-siftsmall}
    
    print_message $GREEN "开始部署所有服务器（数据集: $dataset）"
    
    local success=true
    
    for server_id in "${!SERVERS[@]}"; do
        server_ip="${SERVERS[$server_id]}"
        
        if [[ "$server_ip" == "YOUR_SERVER_"* ]]; then
            print_message $RED "请先在脚本中配置服务器 $server_id 的IP地址"
            continue
        fi
        
        # 检查SSH连接
        if ! check_ssh_connection "$server_ip" "$server_id"; then
            success=false
            continue
        fi
        
        # 同步代码
        if ! sync_code "$server_ip" "$server_id"; then
            success=false
            continue
        fi
        
        # 启动服务器
        if ! start_remote_server "$server_ip" "$server_id" "$dataset"; then
            success=false
            continue
        fi
    done
    
    if $success; then
        print_message $GREEN "所有服务器部署成功！"
    else
        print_message $RED "部分服务器部署失败"
    fi
}

# 主程序
main() {
    # 检查PEM文件是否存在
    if [ ! -f "$PEM_FILE" ]; then
        print_message $RED "错误：找不到密钥文件 $PEM_FILE"
        exit 1
    fi
    
    # 设置密钥文件权限
    chmod 400 "$PEM_FILE"
    
    while true; do
        show_menu
        read -p "请选择操作 (1-8): " choice
        
        case $choice in
            1)
                read -p "请输入数据集名称 (默认: siftsmall): " dataset
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
                read -p "请输入数据集名称 (默认: siftsmall): " dataset
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
                read -p "请输入服务器ID (1-3): " server_id
                if [[ -n "${SERVERS[$server_id]}" ]]; then
                    server_ip="${SERVERS[$server_id]}"
                    if [[ "$server_ip" != "YOUR_SERVER_"* ]]; then
                        read -p "请输入数据集名称 (默认: siftsmall): " dataset
                        dataset=${dataset:-siftsmall}
                        check_ssh_connection "$server_ip" "$server_id" && \
                        sync_code "$server_ip" "$server_id" && \
                        start_remote_server "$server_ip" "$server_id" "$dataset"
                    else
                        print_message $RED "请先配置服务器 $server_id 的IP地址"
                    fi
                else
                    print_message $RED "无效的服务器ID"
                fi
                ;;
            7)
                read -p "请输入服务器ID (1-3): " server_id
                if [[ -n "${SERVERS[$server_id]}" ]]; then
                    server_ip="${SERVERS[$server_id]}"
                    if [[ "$server_ip" != "YOUR_SERVER_"* ]]; then
                        print_message $YELLOW "服务器 $server_id 的最新日志："
                        ssh -i "$PEM_FILE" -o StrictHostKeyChecking=no \
                            ${REMOTE_USER}@${server_ip} \
                            "tail -n 50 ~/test/distributed-deploy/server_${server_id}.log"
                    else
                        print_message $RED "请先配置服务器 $server_id 的IP地址"
                    fi
                else
                    print_message $RED "无效的服务器ID"
                fi
                ;;
            8)
                print_message $GREEN "退出部署工具"
                exit 0
                ;;
            *)
                print_message $RED "无效的选择"
                ;;
        esac
    done
}

# 运行主程序
main