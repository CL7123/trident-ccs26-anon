#!/usr/bin/env python3
"""
简化的安全乘法实现
将所有职责清晰分离的组件放在一个文件中
"""

import sys
import os
import secrets
import time
import logging
import numpy as np
from typing import List, Tuple, Dict, Optional, Any
from dataclasses import dataclass
from collections import Counter
from enum import Enum

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from basic_functionalities import Share, MPC23SSS, DomainConfig, get_config

logger = logging.getLogger(__name__)


# ==================== 数据类型定义 ====================

@dataclass
class TripleShare:
    """Beaver三元组的份额"""
    triple_id: str
    server_id: int
    a_share: Share
    b_share: Share
    c_share: Share


@dataclass 
class ComputationRequest:
    """计算请求"""
    computation_id: str
    x_share: Share
    y_share: Share
    triple_id: str


# ==================== Data Owner: 三元组生成器 ====================

class TripleGenerator:
    """
    Data Owner角色：生成Beaver三元组
    职责：
    1. 生成随机的Beaver三元组
    2. 进行秘密共享
    3. 分发给各服务器
    """
    
    def __init__(self, config: Optional[DomainConfig] = None):
        self.config = config or get_config("laion")
        self.field_size = self.config.prime
        self.mpc = MPC23SSS(self.config)
        self.generated_count = 0
        
    def generate_triples_numpy(self, count: int) -> Dict[int, np.ndarray]:
        """
        生成指定数量的Beaver三元组并分配给服务器（NumPy格式）
        
        Returns:
            Dict[server_id, np.ndarray]: 每个服务器的三元组份额
            数组形状: (count, 3)，每行是 [a_share, b_share, c_share]
        """
        # 初始化服务器数组
        server_arrays = {
            1: np.zeros((count, 3), dtype=np.uint32),
            2: np.zeros((count, 3), dtype=np.uint32),
            3: np.zeros((count, 3), dtype=np.uint32)
        }
        
        # 进度显示设置
        print_interval = max(count // 100, 1000)  # 每1%或每1000个打印一次
        start_time = time.time()
        
        for i in range(count):
            # 生成随机的a, b
            a = secrets.randbelow(self.field_size)
            b = secrets.randbelow(self.field_size)
            c = (a * b) % self.field_size
            
            # 生成秘密份额
            a_shares = self.mpc.share_secret(a)
            b_shares = self.mpc.share_secret(b)
            c_shares = self.mpc.share_secret(c)
            
            # 分配给各服务器
            for server_id in range(1, 4):
                server_arrays[server_id][i] = [
                    a_shares[server_id-1].value,
                    b_shares[server_id-1].value,
                    c_shares[server_id-1].value
                ]
            
            # 打印进度
            if (i + 1) % print_interval == 0 or i == count - 1:
                progress = (i + 1) / count * 100
                elapsed = time.time() - start_time
                rate = (i + 1) / elapsed if elapsed > 0 else 0
                eta = (count - i - 1) / rate if rate > 0 else 0
                print(f"  进度: {i+1:,}/{count:,} ({progress:.1f}%) - "
                      f"速率: {rate:.0f}/秒 - 剩余时间: {eta:.0f}秒")
                
        logger.info(f"Generated {count} Beaver triples in NumPy format")
        return server_arrays
        
    def generate_triples(self, count: int) -> Dict[int, List[TripleShare]]:
        """
        生成指定数量的Beaver三元组并分配给服务器（兼容旧格式）
        
        Returns:
            Dict[server_id, List[TripleShare]]: 每个服务器的三元组份额
        """
        server_shares = {1: [], 2: [], 3: []}
        
        for i in range(count):
            # 生成随机的a, b
            a = secrets.randbelow(self.field_size)
            b = secrets.randbelow(self.field_size)
            c = (a * b) % self.field_size
            
            # 生成秘密份额
            a_shares = self.mpc.share_secret(a)
            b_shares = self.mpc.share_secret(b)
            c_shares = self.mpc.share_secret(c)
            
            # 分配给各服务器
            triple_id = f"triple_{self.generated_count}_{int(time.time()*1000)}"
            self.generated_count += 1
            
            for server_id in range(1, 4):
                triple_share = TripleShare(
                    triple_id=triple_id,
                    server_id=server_id,
                    a_share=a_shares[server_id-1],
                    b_share=b_shares[server_id-1],
                    c_share=c_shares[server_id-1]
                )
                server_shares[server_id].append(triple_share)
                
        logger.info(f"Generated {count} Beaver triples")
        return server_shares


# ==================== Server: 乘法服务器 ====================

class MultiplicationServer:
    """
    服务器角色：执行安全乘法的本地计算
    职责：
    1. 存储自己的三元组份额
    2. 计算e和f的份额
    3. 参与重构
    4. 计算结果份额
    """
    
    def __init__(self, server_id: int, config: Optional[DomainConfig] = None):
        self.server_id = server_id
        self.config = config or get_config("laion")
        self.field_size = self.config.prime
        self.triple_storage: Dict[str, TripleShare] = {}
        self.computation_cache: Dict[str, Any] = {}
        
    def load_triples(self, triples: List[TripleShare]):
        """加载三元组份额"""
        for triple in triples:
            if triple.server_id != self.server_id:
                raise ValueError(f"Triple {triple.triple_id} not for this server")
            self.triple_storage[triple.triple_id] = triple
        logger.info(f"Server {self.server_id} loaded {len(triples)} triples")
        
    def compute_e_f_shares(self, request: ComputationRequest) -> Tuple[Share, Share]:
        """
        计算e和f的份额
        e = x - a
        f = y - b
        """
        if request.triple_id not in self.triple_storage:
            raise ValueError(f"Triple {request.triple_id} not found")
            
        triple = self.triple_storage[request.triple_id]
        
        # 计算e_i = x_i - a_i
        e_value = (request.x_share.value - triple.a_share.value) % self.field_size
        e_share = Share(e_value, self.server_id)
        
        # 计算f_i = y_i - b_i
        f_value = (request.y_share.value - triple.b_share.value) % self.field_size
        f_share = Share(f_value, self.server_id)
        
        # 缓存用于后续计算
        self.computation_cache[request.computation_id] = {
            'triple': triple,
            'request': request,
            'e_share': e_share,
            'f_share': f_share
        }
        
        return e_share, f_share
        
    def compute_result_share(self, computation_id: str, e: int, f: int) -> Share:
        """
        计算结果份额
        z_i = c_i + e*b_i + f*a_i + e*f
        """
        if computation_id not in self.computation_cache:
            raise ValueError(f"Computation {computation_id} not found")
            
        cache = self.computation_cache[computation_id]
        triple = cache['triple']
        
        # z_i = c_i + e*b_i + f*a_i + e*f
        result = triple.c_share.value
        result = (result + e * triple.b_share.value) % self.field_size
        result = (result + f * triple.a_share.value) % self.field_size
        result = (result + e * f) % self.field_size
        
        # 清理缓存
        del self.computation_cache[computation_id]
        
        # 标记三元组已使用
        del self.triple_storage[triple.triple_id]
        
        return Share(result, self.server_id)


class NumpyMultiplicationServer:
    """
    服务器角色：使用NumPy数组存储的高效版本
    """
    
    def __init__(self, server_id: int, config: Optional[DomainConfig] = None,
                 data_dir: str = "~/trident/dataset/triples"):
        self.server_id = server_id
        self.config = config or get_config("laion")
        self.field_size = self.config.prime
        self.triple_array: Optional[np.ndarray] = None
        self.used_count = 0
        self.computation_cache: Dict[str, Any] = {}
        self.data_dir = data_dir
        
        # 自动加载本地三元组
        self._load_local_triples()
        
    def _load_local_triples(self):
        """自动从本地目录加载三元组"""
        import glob
        
        # 本服务器的数据目录
        server_dir = f"{self.data_dir}/server_{self.server_id}"
        
        # 查找所有三元组文件
        triple_files = sorted(glob.glob(f"{server_dir}/triples_*.npy"))
        
        if not triple_files:
            logger.warning(f"Server {self.server_id}: No triple files found in {server_dir}")
            return
            
        # 加载最大的文件（假设文件名包含数量）
        # 例如: triples_10000.npy -> 10000
        latest_file = max(triple_files, 
                         key=lambda f: int(os.path.basename(f).split('_')[1].split('.')[0]))
        
        # 加载三元组
        self.triple_array = np.load(latest_file)
        self.used_count = 0
        
        logger.info(f"Server {self.server_id}: Loaded {len(self.triple_array)} triples from {latest_file}")
        
    def get_next_triple(self) -> Tuple[int, int, int]:
        """获取下一个未使用的三元组"""
        if self.triple_array is None or self.used_count >= len(self.triple_array):
            raise RuntimeError("No more triples available")
            
        triple = self.triple_array[self.used_count]
        self.used_count += 1
        return int(triple[0]), int(triple[1]), int(triple[2])
    
    def get_triple_at(self, index: int) -> Tuple[int, int, int]:
        """获取指定索引的三元组（用于调试）"""
        if self.triple_array is None or index >= len(self.triple_array):
            raise RuntimeError("Triple index out of range")
        triple = self.triple_array[index]
        return int(triple[0]), int(triple[1]), int(triple[2])
        
    def compute_e_f_shares(self, x_value: int, y_value: int, 
                          computation_id: str) -> Tuple[int, int]:
        """
        计算e和f的份额（简化接口）
        """
        a, b, c = self.get_next_triple()
        
        # 计算e_i = x_i - a_i
        e_value = (x_value - a) % self.field_size
        
        # 计算f_i = y_i - b_i
        f_value = (y_value - b) % self.field_size
        
        # 缓存用于后续计算
        self.computation_cache[computation_id] = {
            'a': a, 'b': b, 'c': c
        }
        
        return e_value, f_value
        
    def compute_result_share(self, computation_id: str, e: int, f: int) -> int:
        """
        计算结果份额（简化版本）
        """
        if computation_id not in self.computation_cache:
            raise ValueError(f"Computation {computation_id} not found")
            
        cache = self.computation_cache[computation_id]
        a, b, c = cache['a'], cache['b'], cache['c']
        
        # z_i = c_i + e*b_i + f*a_i + e*f
        result = c
        result = (result + e * b) % self.field_size
        result = (result + f * a) % self.field_size
        result = (result + e * f) % self.field_size
        
        # 清理缓存
        del self.computation_cache[computation_id]
        
        return result


# ==================== Client: 协调客户端 ====================

class MultiplicationClient:
    """
    客户端角色：协调安全乘法协议
    职责：
    1. 发起乘法请求
    2. 收集e,f份额并重构
    3. 协调最终结果计算
    4. 执行恶意检测
    """
    
    def __init__(self, config: Optional[DomainConfig] = None):
        self.config = config or get_config("laion")
        self.field_size = self.config.prime
        self.mpc = MPC23SSS(self.config)
        self.computation_count = 0
        
    def multiply(self, 
                 x_shares: List[Share], 
                 y_shares: List[Share],
                 servers: List[MultiplicationServer],
                 triple_id: str) -> Tuple[List[Share], Optional[int]]:
        """
        执行安全乘法
        
        Returns:
            (result_shares, detected_malicious_server_id)
        """
        if len(x_shares) != 3 or len(y_shares) != 3:
            raise ValueError("Need exactly 3 shares")
            
        computation_id = f"comp_{self.computation_count}_{int(time.time()*1000)}"
        self.computation_count += 1
        
        # Step 1: 发送计算请求给各服务器
        e_f_shares = []
        for i in range(3):
            request = ComputationRequest(
                computation_id=computation_id,
                x_share=x_shares[i],
                y_share=y_shares[i],
                triple_id=triple_id
            )
            e_share, f_share = servers[i].compute_e_f_shares(request)
            e_f_shares.append((e_share, f_share))
            
        # Step 2: 重构e和f
        e_shares = [ef[0] for ef in e_f_shares]
        f_shares = [ef[1] for ef in e_f_shares]
        
        e = self._reconstruct_with_check(e_shares, "e")
        f = self._reconstruct_with_check(f_shares, "f")
        
        # Step 3: 各服务器计算结果份额
        result_shares = []
        for i in range(3):
            result_share = servers[i].compute_result_share(computation_id, e, f)
            result_shares.append(result_share)
            
        # Step 4: 验证结果（恶意检测）
        detected_malicious = self._verify_result(result_shares)
        
        return result_shares, detected_malicious
        
    def _reconstruct_with_check(self, shares: List[Share], label: str) -> int:
        """重构值并检查一致性"""
        # 三种重构方式
        value_12 = self.mpc.reconstruct([shares[0], shares[1]])
        value_13 = self.mpc.reconstruct([shares[0], shares[2]]) 
        value_23 = self.mpc.reconstruct([shares[1], shares[2]])
        
        # 检查一致性
        if value_12 == value_13 == value_23:
            return value_12
            
        # 多数投票
        values = [value_12, value_13, value_23]
        counter = Counter(values)
        most_common = counter.most_common(1)[0]
        
        if most_common[1] >= 2:
            value = most_common[0]
            # 识别可能的恶意方
            if value_12 == value_13 == value:
                logger.warning(f"Server 3 might be malicious on {label}")
            elif value_12 == value_23 == value:
                logger.warning(f"Server 1 might be malicious on {label}")
            elif value_13 == value_23 == value:
                logger.warning(f"Server 2 might be malicious on {label}")
            return value
        else:
            raise RuntimeError(f"No consensus on {label}: {values}")
            
    def _verify_result(self, result_shares: List[Share]) -> Optional[int]:
        """验证最终结果并检测恶意服务器"""
        # 三种重构
        r12 = self.mpc.reconstruct([result_shares[0], result_shares[1]])
        r13 = self.mpc.reconstruct([result_shares[0], result_shares[2]])
        r23 = self.mpc.reconstruct([result_shares[1], result_shares[2]])
        
        # 全部一致，无恶意
        if r12 == r13 == r23:
            return None
            
        # 识别恶意服务器
        if r12 == r13:
            return 3  # Server 3 is malicious
        elif r12 == r23:
            return 1  # Server 1 is malicious
        elif r13 == r23:
            return 2  # Server 2 is malicious
        else:
            raise RuntimeError("Cannot identify malicious server")


# ==================== 简单测试函数 ====================

def test_secure_multiplication():
    """测试安全乘法"""
    print("=== 测试安全乘法协议 ===\n")
    
    config = get_config("laion")
    
    # 1. Data Owner生成三元组
    print("1. Data Owner生成三元组")
    generator = TripleGenerator(config)
    triple_shares = generator.generate_triples(5)  # 生成5个三元组
    
    # 2. 初始化服务器并加载三元组
    print("\n2. 初始化3个服务器")
    servers = []
    for server_id in range(1, 4):
        server = MultiplicationServer(server_id, config)
        server.load_triples(triple_shares[server_id])
        servers.append(server)
        
    # 3. 客户端执行乘法
    print("\n3. 测试乘法: 42 × 37")
    client = MultiplicationClient(config)
    mpc = MPC23SSS(config)
    
    # 生成输入的秘密份额
    x = 42
    y = 37
    x_shares = mpc.share_secret(x)
    y_shares = mpc.share_secret(y)
    
    # 执行安全乘法
    result_shares, malicious = client.multiply(
        x_shares, y_shares, servers, triple_shares[1][0].triple_id
    )
    
    # 重构结果
    result = mpc.reconstruct(result_shares[:2])
    expected = x * y
    
    print(f"\n结果: {result}")
    print(f"预期: {expected}")
    print(f"验证: {'✓ 正确' if result == expected else '✗ 错误'}")
    print(f"检测到恶意服务器: {malicious if malicious else '无'}")
    
    # 4. 测试批量乘法
    print("\n\n4. 测试批量乘法（模拟VDPF查询）")
    n = 10  # 小规模测试
    selector = [0] * n
    selector[3] = 1  # 选择第4个元素
    data = list(range(10, 10+n))
    
    # 计算 Σ(selector[i] × data[i])
    sum_result = 0
    for i in range(n):
        if i < len(triple_shares[1]):  # 确保有足够的三元组
            s_shares = mpc.share_secret(selector[i])
            d_shares = mpc.share_secret(data[i])
            
            r_shares, _ = client.multiply(
                s_shares, d_shares, servers, triple_shares[1][i].triple_id
            )
            
            r = mpc.reconstruct(r_shares[:2])
            sum_result = (sum_result + r) % config.prime
            
    print(f"查询结果: {sum_result}")
    print(f"预期结果: {data[3]}")
    print(f"验证: {'✓ 正确' if sum_result == data[3] else '✗ 错误'}")
    
    print("\n=== 测试完成 ===")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_secure_multiplication()