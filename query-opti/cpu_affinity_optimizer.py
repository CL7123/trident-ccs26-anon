"""
CPU affinity optimization: bind processes to specific physical cores
"""
import os
import multiprocessing

def get_physical_cores():
    """Get physical core list (excluding hyperthreads)"""
    # On AMD EPYC, physical cores are typically 0-15, hyperthreads are 16-31
    return list(range(16))

def set_process_affinity(server_id, process_id, cores_per_server=4):
    """
    Set process CPU affinity

    Args:
        server_id: Server ID (1, 2, 3)
        process_id: Process ID (0, 1, 2, 3)
        cores_per_server: Number of cores allocated per server
    """
    # Calculate which core this process should bind to
    # Server 1: cores 0-3
    # Server 2: cores 4-7
    # Server 3: cores 8-11
    base_core = (server_id - 1) * cores_per_server
    target_core = base_core + process_id

    # Bind to specific core
    pid = os.getpid()
    os.sched_setaffinity(pid, {target_core})

    print(f"[Server {server_id}, Process {process_id}] Bound to core {target_core}")
    
def get_optimal_process_count():
    """Return optimal process count based on CPU configuration"""
    # For 16-core CPU with 3 servers, 4 processes per server is optimal
    return 4

def print_cpu_topology():
    """Print CPU topology structure"""
    print("CPU topology structure (AMD EPYC 7R32):")
    print("L3 cache 0: cores 0-3   -> Server 1")
    print("L3 cache 1: cores 4-7   -> Server 2")
    print("L3 cache 2: cores 8-11  -> Server 3")
    print("L3 cache 3: cores 12-15 -> System/client")
    print("\nRecommendation: Each server uses 4 processes, bound to the same L3 cache domain")