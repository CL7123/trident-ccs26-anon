#!/bin/bash

# [CN]
echo "[CN]..."
sudo ip netns del client 2>/dev/null
sudo ip netns del server1 2>/dev/null
sudo ip netns del server2 2>/dev/null
sudo ip netns del server3 2>/dev/null
sudo ip link del br0 2>/dev/null

# [CN]
echo "[CN]..."
sudo ip netns add client
sudo ip netns add server1
sudo ip netns add server2
sudo ip netns add server3

# [CN]
sudo ip link add br0 type bridge
sudo ip link set br0 up
# [CN]IP[CN]（[CN]）
sudo ip addr add 192.168.50.1/24 dev br0

# [CN]client
echo "configureclient..."
sudo ip link add veth-c type veth peer name veth-c-br
sudo ip link set veth-c netns client
sudo ip netns exec client ip addr add 192.168.50.10/24 dev veth-c
sudo ip netns exec client ip link set veth-c up
sudo ip netns exec client ip link set lo up
# [CN]
sudo ip netns exec client ip route add default via 192.168.50.1 dev veth-c 2>/dev/null || true
sudo ip link set veth-c-br master br0
sudo ip link set veth-c-br up

# [CN]server
for i in 1 2 3; do
    echo "configureserver$i..."
    sudo ip link add veth-s$i type veth peer name veth-s$i-br
    sudo ip link set veth-s$i netns server$i
    sudo ip netns exec server$i ip addr add 192.168.50.2$i/24 dev veth-s$i
    sudo ip netns exec server$i ip link set veth-s$i up
    sudo ip netns exec server$i ip link set lo up
    # [CN]
    sudo ip netns exec server$i ip route add default via 192.168.50.1 dev veth-s$i 2>/dev/null || true
    sudo ip link set veth-s$i-br master br0
    sudo ip link set veth-s$i-br up
done

# [CN] TC [CN]
echo -e "\n[CN]..."

# clientconfigure：[CN]
echo "configureclient[CN]parameters（3Gbps, 10ms[CN]）..."
sudo ip netns exec client tc qdisc add dev veth-c root handle 1: tbf rate 3gbit burst 15mb latency 50ms
sudo ip netns exec client tc qdisc add dev veth-c parent 1:1 handle 10: netem delay 10ms

# serverconfigure：[CN]client[CN]
for i in 1 2 3; do
    dev="veth-s$i"
    echo "configureserver$i[CN]parameters..."
    
    # [CN]HTB[CN]
    sudo ip netns exec server$i tc qdisc add dev $dev root handle 1: htb default 30
    
    # [CN]：[CN]client[CN]（[CN]），[CN]server[CN]（[CN]）
    # [CN]1:10 - [CN]client[CN]（3Gbps[CN]+10ms[CN]）
    sudo ip netns exec server$i tc class add dev $dev parent 1: classid 1:10 htb rate 3gbit
    sudo ip netns exec server$i tc qdisc add dev $dev parent 1:10 handle 10: netem delay 10ms
    
    # [CN]1:30 - [CN]，server[CN]（[CN]）
    sudo ip netns exec server$i tc class add dev $dev parent 1: classid 1:30 htb rate 100gbit
    
    # [CN]：[CN]client192.168.50.10[CN]
    sudo ip netns exec server$i tc filter add dev $dev protocol ip parent 1:0 prio 1 u32 \
        match ip dst 192.168.50.10/32 flowid 1:10
done

# [CN]configure
echo -e "\n[CN]configure:"
echo "1. [CN]test:"
for i in 1 2 3; do
    echo -n "  Client -> Server$i: "
    sudo ip netns exec client ping -c 1 -W 1 192.168.50.2$i >/dev/null 2>&1 && echo "OK" || echo "FAIL"
done

echo -e "\n2. RTTtest:"
echo "  Client -> Server1（[CN]20ms）:"
sudo ip netns exec client ping -c 5 192.168.50.21 | grep -E "min/avg/max"
echo "  Server1 -> Server2（[CN]<1ms，server[CN]）:"
sudo ip netns exec server1 ping -c 5 192.168.50.22 | grep -E "min/avg/max"
echo "  Server2 -> Server3（[CN]<1ms，server[CN]）:"
sudo ip netns exec server2 ping -c 5 192.168.50.23 | grep -E "min/avg/max"

echo -e "\n3. TC[CN]:"
echo "  clientTCconfigure:"
sudo ip netns exec client tc qdisc show dev veth-c
echo "  server1 TCconfigure:"
sudo ip netns exec server1 tc class show dev veth-s1

echo -e "\n=== [CN]configure[CN] ==="
echo "[CN]parameters："
echo "  - Client <-> Server: 3 Gbps[CN]，20ms RTT"
echo "  - Server <-> Server: [CN]（[CN]）"
