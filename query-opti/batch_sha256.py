"""
Batch SHA-256 computation optimization
Using OpenSSL EVP interface or hardware acceleration
"""
import numpy as np
import hashlib
from concurrent.futures import ThreadPoolExecutor
import multiprocessing


class BatchSHA256:
    """Batch SHA-256 calculator"""

    def __init__(self, num_threads=4):
        self.num_threads = num_threads

    def compute_batch_parallel(self, inputs: list) -> list:
        """Compute multiple SHA-256 in parallel"""
        # Split input to multiple threads
        chunk_size = max(1, len(inputs) // self.num_threads)
        chunks = [inputs[i:i+chunk_size] for i in range(0, len(inputs), chunk_size)]

        def compute_chunk(chunk):
            results = []
            for data in chunk:
                h = hashlib.sha256()
                h.update(data)
                results.append(h.digest())
            return results

        # Use thread pool for parallel computation
        with ThreadPoolExecutor(max_workers=self.num_threads) as executor:
            chunk_results = list(executor.map(compute_chunk, chunks))

        # Merge results
        results = []
        for chunk_result in chunk_results:
            results.extend(chunk_result)

        return results

    def compute_batch_vectorized(self, seeds: list, suffixes: list) -> list:
        """Vectorized batch computation (simulated)"""
        # Prepare all inputs
        inputs = []
        for seed in seeds:
            for suffix in suffixes:
                inputs.append(seed + suffix)

        # Batch computation
        return self.compute_batch_parallel(inputs)


# Try to import faster SHA implementation
try:
    import cryptography.hazmat.primitives.hashes as crypto_hashes
    from cryptography.hazmat.backends import default_backend

    class FastBatchSHA256(BatchSHA256):
        """Faster SHA-256 implementation using cryptography library"""

        def compute_single(self, data: bytes) -> bytes:
            """Compute single SHA-256 using cryptography"""
            digest = crypto_hashes.Hash(crypto_hashes.SHA256(), backend=default_backend())
            digest.update(data)
            return digest.finalize()

        def compute_batch_parallel(self, inputs: list) -> list:
            """Parallel batch computation using cryptography"""
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
    FastBatchSHA256 = BatchSHA256  # Fallback to standard implementation
