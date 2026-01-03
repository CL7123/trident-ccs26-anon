#!/bin/bash

# Clean up old environment
echo "Cleaning up old environment..."
sudo ip netns del client 2>/dev/null
sudo ip netns del server1 2>/dev/null
sudo ip netns del server2 2>/dev/null
sudo ip netns del server3 2>/dev/null
sudo ip link del br0 2>/dev/null

# Create network topology
echo "Creating network namespaces..."
sudo ip netns add client
sudo ip netns add server1
sudo ip netns add server2
sudo ip netns add server3

# Create bridge
sudo ip link add br0 type bridge
sudo ip link set br0 up
# Assign IP address to bridge (as gateway)
sudo ip addr add 192.168.50.1/24 dev br0

# Configure client
echo "Configuring client..."
sudo ip link add veth-c type veth peer name veth-c-br
sudo ip link set veth-c netns client
sudo ip netns exec client ip addr add 192.168.50.10/24 dev veth-c
sudo ip netns exec client ip link set veth-c up
sudo ip netns exec client ip link set lo up
# Add default route
sudo ip netns exec client ip route add default via 192.168.50.1 dev veth-c 2>/dev/null || true
sudo ip link set veth-c-br master br0
sudo ip link set veth-c-br up

# Configure servers
for i in 1 2 3; do
    echo "Configuring server$i..."
    sudo ip link add veth-s$i type veth peer name veth-s$i-br
    sudo ip link set veth-s$i netns server$i
    sudo ip netns exec server$i ip addr add 192.168.50.2$i/24 dev veth-s$i
    sudo ip netns exec server$i ip link set veth-s$i up
    sudo ip netns exec server$i ip link set lo up
    # Add default route
    sudo ip netns exec server$i ip route add default via 192.168.50.1 dev veth-s$i 2>/dev/null || true
    sudo ip link set veth-s$i-br master br0
    sudo ip link set veth-s$i-br up
done

# Apply differentiated TC rules
echo -e "\nApplying differentiated traffic control rules..."

# Client configuration: apply limits to all outbound traffic
echo "Configuring client network parameters (3Gbps, 10ms latency)..."
sudo ip netns exec client tc qdisc add dev veth-c root handle 1: tbf rate 3gbit burst 15mb latency 50ms
sudo ip netns exec client tc qdisc add dev veth-c parent 1:1 handle 10: netem delay 10ms

# Server configuration: apply limits only to traffic destined for client
for i in 1 2 3; do
    dev="veth-s$i"
    echo "Configuring server$i network parameters..."

    # Use HTB to create classification queue
    sudo ip netns exec server$i tc qdisc add dev $dev root handle 1: htb default 30

    # Create two classes: one for traffic to client (limited), one for traffic to other servers (unlimited)
    # Class 1:10 - Traffic to client (3Gbps limit + 10ms latency)
    sudo ip netns exec server$i tc class add dev $dev parent 1: classid 1:10 htb rate 3gbit
    sudo ip netns exec server$i tc qdisc add dev $dev parent 1:10 handle 10: netem delay 10ms

    # Class 1:30 - Default class, inter-server traffic (unlimited)
    sudo ip netns exec server$i tc class add dev $dev parent 1: classid 1:30 htb rate 100gbit

    # Add filter: traffic destined for client 192.168.50.10 uses limited channel
    sudo ip netns exec server$i tc filter add dev $dev protocol ip parent 1:0 prio 1 u32 \
        match ip dst 192.168.50.10/32 flowid 1:10
done

# Verify configuration
echo -e "\nVerifying network configuration:"
echo "1. Connectivity test:"
for i in 1 2 3; do
    echo -n "  Client -> Server$i: "
    sudo ip netns exec client ping -c 1 -W 1 192.168.50.2$i >/dev/null 2>&1 && echo "OK" || echo "FAIL"
done

echo -e "\n2. RTT test:"
echo "  Client -> Server1 (should be ~20ms):"
sudo ip netns exec client ping -c 5 192.168.50.21 | grep -E "min/avg/max"
echo "  Server1 -> Server2 (should be <1ms, no delay between servers):"
sudo ip netns exec server1 ping -c 5 192.168.50.22 | grep -E "min/avg/max"
echo "  Server2 -> Server3 (should be <1ms, no delay between servers):"
sudo ip netns exec server2 ping -c 5 192.168.50.23 | grep -E "min/avg/max"

echo -e "\n3. View TC rules:"
echo "  Client TC configuration:"
sudo ip netns exec client tc qdisc show dev veth-c
echo "  Server1 TC configuration:"
sudo ip netns exec server1 tc class show dev veth-s1

echo -e "\n=== Network configuration completed ==="
echo "Network parameters:"
echo "  - Client <-> Server: 3 Gbps bandwidth, 20ms RTT"
echo "  - Server <-> Server: Unlimited (same data center)"
