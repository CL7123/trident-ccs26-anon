"""
批量SHA-256计算优化
使用OpenSSL的EVP接口或硬件加速
"""
import numpy as np
import hashlib
from concurrent.futures import ThreadPoolExecutor
import multiprocessing


class BatchSHA256:
    """批量SHA-256计算器"""
    
    def __init__(self, num_threads=4):
        self.num_threads = num_threads
        
    def compute_batch_parallel(self, inputs: list) -> list:
        """并行计算多个SHA-256"""
        # 分割输入到多个线程
        chunk_size = max(1, len(inputs) // self.num_threads)
        chunks = [inputs[i:i+chunk_size] for i in range(0, len(inputs), chunk_size)]
        
        def compute_chunk(chunk):
            results = []
            for data in chunk:
                h = hashlib.sha256()
                h.update(data)
                results.append(h.digest())
            return results
        
        # 使用线程池并行计算
        with ThreadPoolExecutor(max_workers=self.num_threads) as executor:
            chunk_results = list(executor.map(compute_chunk, chunks))
        
        # 合并结果
        results = []
        for chunk_result in chunk_results:
            results.extend(chunk_result)
        
        return results
    
    def compute_batch_vectorized(self, seeds: list, suffixes: list) -> list:
        """向量化批量计算（模拟）"""
        # 准备所有输入
        inputs = []
        for seed in seeds:
            for suffix in suffixes:
                inputs.append(seed + suffix)
        
        # 批量计算
        return self.compute_batch_parallel(inputs)


# 尝试导入更快的SHA实现
try:
    import cryptography.hazmat.primitives.hashes as crypto_hashes
    from cryptography.hazmat.backends import default_backend
    
    class FastBatchSHA256(BatchSHA256):
        """使用cryptography库的更快SHA-256实现"""
        
        def compute_single(self, data: bytes) -> bytes:
            """使用cryptography计算单个SHA-256"""
            digest = crypto_hashes.Hash(crypto_hashes.SHA256(), backend=default_backend())
            digest.update(data)
            return digest.finalize()
        
        def compute_batch_parallel(self, inputs: list) -> list:
            """使用cryptography的并行批量计算"""
            chunk_size = max(1, len(inputs) // self.num_threads)
            chunks = [inputs[i:i+chunk_size] for i in range(0, len(inputs), chunk_size)]
            
            def compute_chunk(chunk):
                results = []
                backend = default_backend()
                for data in chunk:
                    digest = crypto_hashes.Hash(crypto_hashes.SHA256(), backend=backend)
                    digest.update(data)
                    results.append(digest.finalize())
                return results
            
            with ThreadPoolExecutor(max_workers=self.num_threads) as executor:
                chunk_results = list(executor.map(compute_chunk, chunks))
            
            results = []
            for chunk_result in chunk_results:
                results.extend(chunk_result)
            
            return results
            
except ImportError:
    FastBatchSHA256 = BatchSHA256  # 回退到标准实现