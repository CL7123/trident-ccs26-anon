"""
CPU[CN]：[CN]bind[CN]
"""
import os
import multiprocessing

def get_physical_cores():
    """[CN]（[CN]）"""
    # [CN]AMD EPYC[CN]，[CN]0-15，[CN]16-31
    return list(range(16))

def set_process_affinity(server_id, process_id, cores_per_server=4):
    """
    [CN]CPU[CN]
    
    Args:
        server_id: [CN]ID (1, 2, 3)
        process_id: [CN]ID (0, 1, 2, 3)
        cores_per_server: cores allocated per server
    """
    # calculate[CN]bind[CN]
    # Server 1: [CN] 0-3
    # Server 2: [CN] 4-7
    # Server 3: [CN] 8-11
    base_core = (server_id - 1) * cores_per_server
    target_core = base_core + process_id
    
    # bind[CN]
    pid = os.getpid()
    os.sched_setaffinity(pid, {target_core})
    
    print(f"[Server {server_id}, Process {process_id}] bind[CN] {target_core}")
    
def get_optimal_process_count():
    """[CN]CPU[CN]return[CN]"""
    # [CN]16[CN]CPU，3[CN]server，[CN]server 4[CN]
    return 4

def print_cpu_topology():
    """printCPU[CN]"""
    print("CPU[CN]（AMD EPYC 7R32）:")
    print("L3[CN]0: [CN]0-3   → Server 1")
    print("L3[CN]1: [CN]4-7   → Server 2") 
    print("L3[CN]2: [CN]8-11  → Server 3")
    print("L3[CN]3: [CN]12-15 → [CN]/[CN]")
    print("\n[CN]：[CN]server[CN]4[CN]，bind[CN]L3[CN]")