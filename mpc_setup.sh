#!/bin/bash

# 清理旧环境
echo "清理旧环境..."
sudo ip netns del client 2>/dev/null
sudo ip netns del server1 2>/dev/null
sudo ip netns del server2 2>/dev/null
sudo ip netns del server3 2>/dev/null
sudo ip link del br0 2>/dev/null

# 创建网络拓扑
echo "创建网络命名空间..."
sudo ip netns add client
sudo ip netns add server1
sudo ip netns add server2
sudo ip netns add server3

# 创建网桥
sudo ip link add br0 type bridge
sudo ip link set br0 up
# 给网桥分配IP地址（作为网关）
sudo ip addr add 192.168.50.1/24 dev br0

# 设置客户端
echo "配置客户端..."
sudo ip link add veth-c type veth peer name veth-c-br
sudo ip link set veth-c netns client
sudo ip netns exec client ip addr add 192.168.50.10/24 dev veth-c
sudo ip netns exec client ip link set veth-c up
sudo ip netns exec client ip link set lo up
# 添加默认路由
sudo ip netns exec client ip route add default via 192.168.50.1 dev veth-c 2>/dev/null || true
sudo ip link set veth-c-br master br0
sudo ip link set veth-c-br up

# 设置服务器
for i in 1 2 3; do
    echo "配置服务器$i..."
    sudo ip link add veth-s$i type veth peer name veth-s$i-br
    sudo ip link set veth-s$i netns server$i
    sudo ip netns exec server$i ip addr add 192.168.50.2$i/24 dev veth-s$i
    sudo ip netns exec server$i ip link set veth-s$i up
    sudo ip netns exec server$i ip link set lo up
    # 添加默认路由
    sudo ip netns exec server$i ip route add default via 192.168.50.1 dev veth-s$i 2>/dev/null || true
    sudo ip link set veth-s$i-br master br0
    sudo ip link set veth-s$i-br up
done

# 应用差异化的 TC 规则
echo -e "\n应用差异化流量控制规则..."

# 客户端配置：对所有出站流量应用限制
echo "配置客户端网络参数（3Gbps, 10ms延迟）..."
sudo ip netns exec client tc qdisc add dev veth-c root handle 1: tbf rate 3gbit burst 15mb latency 50ms
sudo ip netns exec client tc qdisc add dev veth-c parent 1:1 handle 10: netem delay 10ms

# 服务器配置：仅对到客户端的流量应用限制
for i in 1 2 3; do
    dev="veth-s$i"
    echo "配置服务器$i网络参数..."
    
    # 使用HTB创建分类队列
    sudo ip netns exec server$i tc qdisc add dev $dev root handle 1: htb default 30
    
    # 创建两个类：一个用于到客户端的流量（限制），一个用于到其他服务器的流量（不限制）
    # 类1:10 - 到客户端的流量（3Gbps限制+10ms延迟）
    sudo ip netns exec server$i tc class add dev $dev parent 1: classid 1:10 htb rate 3gbit
    sudo ip netns exec server$i tc qdisc add dev $dev parent 1:10 handle 10: netem delay 10ms
    
    # 类1:30 - 默认类，服务器间流量（无限制）
    sudo ip netns exec server$i tc class add dev $dev parent 1: classid 1:30 htb rate 100gbit
    
    # 添加过滤器：到客户端192.168.50.10的流量走限制通道
    sudo ip netns exec server$i tc filter add dev $dev protocol ip parent 1:0 prio 1 u32 \
        match ip dst 192.168.50.10/32 flowid 1:10
done

# 验证配置
echo -e "\n验证网络配置:"
echo "1. 连通性测试:"
for i in 1 2 3; do
    echo -n "  Client -> Server$i: "
    sudo ip netns exec client ping -c 1 -W 1 192.168.50.2$i >/dev/null 2>&1 && echo "OK" || echo "FAIL"
done

echo -e "\n2. RTT测试:"
echo "  Client -> Server1（应该约20ms）:"
sudo ip netns exec client ping -c 5 192.168.50.21 | grep -E "min/avg/max"
echo "  Server1 -> Server2（应该<1ms，服务器间无延迟）:"
sudo ip netns exec server1 ping -c 5 192.168.50.22 | grep -E "min/avg/max"
echo "  Server2 -> Server3（应该<1ms，服务器间无延迟）:"
sudo ip netns exec server2 ping -c 5 192.168.50.23 | grep -E "min/avg/max"

echo -e "\n3. TC规则查看:"
echo "  客户端TC配置:"
sudo ip netns exec client tc qdisc show dev veth-c
echo "  服务器1 TC配置:"
sudo ip netns exec server1 tc class show dev veth-s1

echo -e "\n=== 网络配置完成 ==="
echo "网络参数："
echo "  - Client <-> Server: 3 Gbps带宽，20ms RTT"
echo "  - Server <-> Server: 无限制（同数据中心）"
