#!/usr/bin/env python3
"""
Detailed Performance Profiling Script

This script executes a single query and collects comprehensive performance metrics including:
- Client-side DPF key generation time
- Per-server RTT (round-trip time)
- Server-side detailed timing for all 4 phases
- Server-to-server communication details (data volume, timing, speed)

Output: JSON file with complete performance breakdown
"""

import sys
import os
import json
import argparse
import time
from datetime import datetime

# Add project paths
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from client import DistributedClient


def format_bytes(bytes_count):
    """Format bytes to human-readable string"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if bytes_count < 1024.0:
            return f"{bytes_count:.2f} {unit}"
        bytes_count /= 1024.0
    return f"{bytes_count:.2f} TB"


def print_profiling_summary(profiling_data):
    """Print a human-readable summary of profiling data"""
    print("\n" + "="*80)
    print("DETAILED PERFORMANCE PROFILING SUMMARY")
    print("="*80)

    # Query info
    print(f"\nQuery ID: {profiling_data['query_id']}")
    print(f"Node ID: {profiling_data['node_id']}")
    print(f"Success: {profiling_data['success']}")

    # Client metrics
    print("\n" + "-"*80)
    print("CLIENT METRICS")
    print("-"*80)

    dpf_gen = profiling_data['client_metrics']['dpf_key_generation']
    print(f"\nDPF Key Generation:")
    print(f"  Duration: {dpf_gen['duration_ms']:.2f} ms")

    print(f"\nClient-Server Communication:")
    per_server = profiling_data['client_metrics']['per_server_communication']
    for server_id in sorted(per_server.keys()):
        metrics = per_server[server_id]
        if metrics.get('success'):
            print(f"  Client → Server {server_id}:")
            print(f"    RTT: {metrics['rtt_ms']:.2f} ms")
            print(f"    Request sent: {format_bytes(metrics.get('bytes_sent', 0))} in {metrics.get('send_duration_ms', 0):.2f} ms")
            if metrics.get('upload_speed_mbps'):
                print(f"      Upload speed: {metrics['upload_speed_mbps']:.2f} Mbps")
            print(f"    Response received: {format_bytes(metrics.get('bytes_received', 0))} in {metrics.get('receive_duration_ms', 0):.2f} ms")
            if metrics.get('download_speed_mbps'):
                print(f"      Download speed: {metrics['download_speed_mbps']:.2f} Mbps")
        else:
            print(f"  Server {server_id}: FAILED - {metrics.get('error', 'Unknown error')}")

    # Total client-server communication
    total_sent = sum(m.get('bytes_sent', 0) for m in per_server.values() if m.get('success'))
    total_received = sum(m.get('bytes_received', 0) for m in per_server.values() if m.get('success'))
    if total_sent > 0 or total_received > 0:
        print(f"\n  Total Client-Server Traffic:")
        print(f"    Uploaded: {format_bytes(total_sent)}")
        print(f"    Downloaded: {format_bytes(total_received)}")
        print(f"    Total: {format_bytes(total_sent + total_received)}")

    # Server metrics
    if profiling_data['success']:
        print("\n" + "-"*80)
        print("SERVER METRICS")
        print("-"*80)

        for server_id in sorted(profiling_data['server_responses'].keys()):
            server_data = profiling_data['server_responses'][server_id]
            timing = server_data['timing']

            print(f"\nServer {server_id}:")
            print(f"  Phase 1 (VDPF Evaluation): {timing.get('phase1_time', 0):.2f} ms")
            print(f"  Phase 2 (e/f Computation): {timing.get('phase2_time', 0):.2f} ms")
            print(f"  Phase 3 (Data Exchange):   {timing.get('phase3_time', 0):.2f} ms")
            print(f"  Phase 4 (Reconstruction):  {timing.get('phase4_time', 0):.2f} ms")
            print(f"  Total Server Time:         {timing.get('total', 0):.2f} ms")
            print(f"  Triples Used:              {timing.get('triples_used', 0)}")

            # Phase 3 detailed breakdown
            detailed_profiling = server_data.get('detailed_profiling', {})
            if detailed_profiling:
                print(f"\n  Phase 3 Detailed Breakdown:")

                # Send details
                phase3_send = detailed_profiling.get('phase3_send', {})
                if phase3_send:
                    print(f"    Sending to other servers:")
                    for target_id in sorted(phase3_send.keys()):
                        send_info = phase3_send[target_id]
                        if send_info.get('success'):
                            print(f"      → Server {target_id}:")
                            print(f"          Duration: {send_info['duration_ms']:.2f} ms")
                            print(f"          Data Sent: {format_bytes(send_info['data_sent_bytes'])}")
                            print(f"          Speed: {send_info['speed_mbps']:.2f} Mbps")

                # Receive details
                phase3_receive = detailed_profiling.get('phase3_receive', {})
                if phase3_receive:
                    print(f"    Receiving from other servers:")
                    for source_id in sorted(phase3_receive.keys()):
                        recv_info = phase3_receive[source_id]
                        print(f"      ← Server {source_id}:")
                        print(f"          Duration: {recv_info['duration_ms']:.2f} ms")
                        print(f"          Data Received: {format_bytes(recv_info['data_received_bytes'])}")
                        print(f"          Speed: {recv_info['speed_mbps']:.2f} Mbps")

    # Communication matrix summary
    if profiling_data['success'] and len(profiling_data['server_responses']) >= 2:
        print("\n" + "-"*80)
        print("COMMUNICATION MATRIX SUMMARY")
        print("-"*80)

        # Collect all server-to-server transfers
        transfers = []
        for server_id, server_data in profiling_data['server_responses'].items():
            detailed = server_data.get('detailed_profiling', {})
            phase3_send = detailed.get('phase3_send', {})
            for target_id, send_info in phase3_send.items():
                if send_info.get('success'):
                    transfers.append({
                        'from': server_id,
                        'to': target_id,
                        'duration_ms': send_info['duration_ms'],
                        'bytes': send_info['data_sent_bytes'],
                        'speed_mbps': send_info['speed_mbps']
                    })

        if transfers:
            print("\nServer-to-Server Transfers:")
            for transfer in sorted(transfers, key=lambda x: (x['from'], x['to'])):
                print(f"  Server {transfer['from']} → Server {transfer['to']}:")
                print(f"    {format_bytes(transfer['bytes'])} in {transfer['duration_ms']:.2f} ms ({transfer['speed_mbps']:.2f} Mbps)")

            # Calculate totals
            total_bytes = sum(t['bytes'] for t in transfers)
            avg_duration = sum(t['duration_ms'] for t in transfers) / len(transfers)
            avg_speed = sum(t['speed_mbps'] for t in transfers) / len(transfers)

            print(f"\n  Total Data Transferred (all pairs): {format_bytes(total_bytes)}")
            print(f"  Average Transfer Duration: {avg_duration:.2f} ms")
            print(f"  Average Transfer Speed: {avg_speed:.2f} Mbps")

    print("\n" + "="*80)


def save_profiling_data(profiling_data, output_file):
    """Save profiling data to JSON file"""
    # Add timestamp
    profiling_data['timestamp'] = datetime.now().isoformat()

    # Save to file
    with open(output_file, 'w') as f:
        json.dump(profiling_data, f, indent=2)

    print(f"\nProfiling data saved to: {output_file}")
    print(f"File size: {os.path.getsize(output_file)} bytes")


def main():
    parser = argparse.ArgumentParser(description='Detailed Performance Profiling')
    parser.add_argument('--dataset', default='siftsmall', help='Dataset name (default: siftsmall)')
    parser.add_argument('--node-id', type=int, default=1723, help='Node ID to query (default: 1723)')
    parser.add_argument('--output', default='profiling_results.json', help='Output JSON file (default: profiling_results.json)')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose output')

    args = parser.parse_args()

    print("="*80)
    print("DETAILED PERFORMANCE PROFILING")
    print("="*80)
    print(f"\nDataset: {args.dataset}")
    print(f"Node ID: {args.node_id}")
    print(f"Output file: {args.output}")
    print()

    # Create client
    print("Initializing client...")
    client = DistributedClient(dataset=args.dataset)

    # Connect to servers
    print("Connecting to servers...")
    if not client.connect_to_servers():
        print("ERROR: Failed to connect to servers")
        return 1

    # Get server status
    if args.verbose:
        print("\nServer Status:")
        client.get_server_status()

    # Execute profiling query
    print("\n" + "="*80)
    print("EXECUTING PROFILING QUERY")
    print("="*80)

    profiling_data = client.detailed_query_with_profiling(node_id=args.node_id)

    # Print summary
    print_profiling_summary(profiling_data)

    # Save to file
    save_profiling_data(profiling_data, args.output)

    # Disconnect
    client.disconnect_from_servers()

    if profiling_data['success']:
        print("\n✓ Profiling completed successfully!")
        return 0
    else:
        print("\n✗ Profiling failed!")
        return 1


if __name__ == '__main__':
    sys.exit(main())
