"""
CPU亲和性优化：将进程绑定到特定物理核心
"""
import os
import multiprocessing

def get_physical_cores():
    """获取物理核心列表（排除超线程）"""
    # 在AMD EPYC上，物理核心通常是0-15，超线程是16-31
    return list(range(16))

def set_process_affinity(server_id, process_id, cores_per_server=4):
    """
    设置进程CPU亲和性
    
    Args:
        server_id: 服务器ID (1, 2, 3)
        process_id: 进程ID (0, 1, 2, 3)
        cores_per_server: 每个服务器分配的核心数
    """
    # 计算该进程应该绑定的核心
    # Server 1: 核心 0-3
    # Server 2: 核心 4-7
    # Server 3: 核心 8-11
    base_core = (server_id - 1) * cores_per_server
    target_core = base_core + process_id
    
    # 绑定到特定核心
    pid = os.getpid()
    os.sched_setaffinity(pid, {target_core})
    
    print(f"[Server {server_id}, Process {process_id}] 绑定到核心 {target_core}")
    
def get_optimal_process_count():
    """根据CPU配置返回最优进程数"""
    # 对于16核CPU，3个server，每个server 4个进程最优
    return 4

def print_cpu_topology():
    """打印CPU拓扑结构"""
    print("CPU拓扑结构（AMD EPYC 7R32）:")
    print("L3缓存0: 核心0-3   → Server 1")
    print("L3缓存1: 核心4-7   → Server 2") 
    print("L3缓存2: 核心8-11  → Server 3")
    print("L3缓存3: 核心12-15 → 系统/客户端")
    print("\n建议：每个server使用4个进程，绑定到同一L3缓存域")