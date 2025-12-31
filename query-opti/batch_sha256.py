"""
[CN]SHA-256calculate[CN]
[CN]OpenSSL[CN]EVP[CN]
"""
import numpy as np
import hashlib
from concurrent.futures import ThreadPoolExecutor
import multiprocessing


class BatchSHA256:
    """[CN]SHA-256calculate[CN]"""
    
    def __init__(self, num_threads=4):
        self.num_threads = num_threads
        
    def compute_batch_parallel(self, inputs: list) -> list:
        """[CN]calculate[CN]SHA-256"""
        # [CN]
        chunk_size = max(1, len(inputs) // self.num_threads)
        chunks = [inputs[i:i+chunk_size] for i in range(0, len(inputs), chunk_size)]
        
        def compute_chunk(chunk):
            results = []
            for data in chunk:
                h = hashlib.sha256()
                h.update(data)
                results.append(h.digest())
            return results
        
        # [CN]calculate
        with ThreadPoolExecutor(max_workers=self.num_threads) as executor:
            chunk_results = list(executor.map(compute_chunk, chunks))
        
        # [CN]
        results = []
        for chunk_result in chunk_results:
            results.extend(chunk_result)
        
        return results
    
    def compute_batch_vectorized(self, seeds: list, suffixes: list) -> list:
        """[CN]calculate（[CN]）"""
        # [CN]
        inputs = []
        for seed in seeds:
            for suffix in suffixes:
                inputs.append(seed + suffix)
        
        # [CN]calculate
        return self.compute_batch_parallel(inputs)


# [CN]SHA[CN]
try:
    import cryptography.hazmat.primitives.hashes as crypto_hashes
    from cryptography.hazmat.backends import default_backend
    
    class FastBatchSHA256(BatchSHA256):
        """[CN]cryptography[CN]SHA-256[CN]"""
        
        def compute_single(self, data: bytes) -> bytes:
            """[CN]cryptographycalculate[CN]SHA-256"""
            digest = crypto_hashes.Hash(crypto_hashes.SHA256(), backend=default_backend())
            digest.update(data)
            return digest.finalize()
        
        def compute_batch_parallel(self, inputs: list) -> list:
            """[CN]cryptography[CN]calculate"""
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
    FastBatchSHA256 = BatchSHA256  # [CN]