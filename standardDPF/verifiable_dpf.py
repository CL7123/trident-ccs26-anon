#!/usr/bin/env python3
"""
Verifiable DPF (VDPF) Implementation
Based on "Lightweight, Maliciously Secure Verifiable Function Secret Sharing"
by de Castro and Polychroniadou (2021)
"""

import hashlib
from typing import Tuple, List
from dataclasses import dataclass
from standard_dpf import StandardDPF, DPFKey


@dataclass
class VDPFKey(DPFKey):
    """VDPF key structure extending standard DPF key"""
    cs: List[bytes]  # Correction strings for verification (4 x 16 bytes)


@dataclass
class VDPFProof:
    """VDPF proof structure"""
    hash_value: bytes  # 32-byte SHA-256 hash


class MMOHash:
    """
    Matyas-Meyer-Oseas (MMO) hash function implementation
    Used for verification in VDPF
    """
    
    def __init__(self):
        self.block_size = 16  # 128 bits
        
    def _aes_encrypt(self, key: bytes, plaintext: bytes) -> bytes:
        """Simplified AES encryption using SHA-256 as PRF"""
        # In production, use proper AES implementation
        h = hashlib.sha256()
        h.update(key)
        h.update(plaintext)
        return h.digest()[:16]  # Take first 128 bits
    
    def hash_2to4(self, input1: bytes, input2: bytes) -> List[bytes]:
        """
        MMO hash function: 2 blocks to 4 blocks
        H(x1, x2) -> (y1, y2, y3, y4)
        """
        # Ensure inputs are 16 bytes
        assert len(input1) == 16 and len(input2) == 16
        
        output = []
        combined = input1 + input2
        
        # Generate 4 output blocks
        for i in range(4):
            # Use different constants for each output block
            h = hashlib.sha256()
            h.update(combined)
            h.update(i.to_bytes(1, 'big'))
            block = h.digest()[:16]
            output.append(block)
            
        return output
    
    def hash_4to4(self, inputs: List[bytes]) -> List[bytes]:
        """
        MMO hash function: 4 blocks to 4 blocks
        H'(x1, x2, x3, x4) -> (y1, y2, y3, y4)
        """
        assert len(inputs) == 4
        assert all(len(inp) == 16 for inp in inputs)
        
        output = []
        combined = b''.join(inputs)
        
        # Generate 4 output blocks
        for i in range(4):
            h = hashlib.sha256()
            h.update(combined)
            h.update((i + 4).to_bytes(1, 'big'))  # Different constant from hash_2to4
            block = h.digest()[:16]
            output.append(block)
            
        return output


class VerifiableDPF(StandardDPF):
    """
    Verifiable DPF implementation
    Extends standard DPF with verification capabilities
    """
    
    def __init__(self, security_param: int = 128):
        super().__init__(security_param)
        self.mmo = MMOHash()
        
    def gen(self, domain_bits: int, index: int, beta: int = 1) -> Tuple[VDPFKey, VDPFKey]:
        """
        Generate VDPF keys with verification support
        
        Args:
            domain_bits: Number of bits in the domain
            index: Target index where output should be non-zero
            beta: The value that the two parties' outputs should XOR to at the target index (default: 1)
        """
        while True:
            # Generate standard DPF keys
            dpf_key0, dpf_key1 = super().gen(domain_bits, index, beta)
            
            # Get final seeds for verification
            seed0 = self._get_final_seed(dpf_key0, index)
            seed1 = self._get_final_seed(dpf_key1, index)
            
            # Check LSB condition
            bit0 = seed0[0] & 1
            bit1 = seed1[0] & 1
            
            if bit0 != bit1:
                # Verification setup successful
                break
        
        # Compute verification commitments
        index_bytes = index.to_bytes(16, 'big')
        
        # pi0 = H(index || seed0)
        pi0 = self.mmo.hash_2to4(index_bytes, seed0)
        
        # pi1 = H(index || seed1)
        pi1 = self.mmo.hash_2to4(index_bytes, seed1)
        
        # Compute correction strings: cs = pi0 ⊕ pi1
        cs = []
        for i in range(4):
            cs_block = bytes(a ^ b for a, b in zip(pi0[i], pi1[i]))
            cs.append(cs_block)
        
        # Create VDPF keys - extending the standard DPF keys
        vdpf_key0 = VDPFKey(
            party_id=dpf_key0.party_id,
            initial_seed=dpf_key0.initial_seed,
            initial_bit=dpf_key0.initial_bit,
            correction_words=dpf_key0.correction_words,
            last_cw=dpf_key0.last_cw,
            domain_bits=dpf_key0.domain_bits,
            cs=cs
        )
        
        vdpf_key1 = VDPFKey(
            party_id=dpf_key1.party_id,
            initial_seed=dpf_key1.initial_seed,
            initial_bit=dpf_key1.initial_bit,
            correction_words=dpf_key1.correction_words,
            last_cw=dpf_key1.last_cw,
            domain_bits=dpf_key1.domain_bits,
            cs=cs
        )
        
        return vdpf_key0, vdpf_key1
    
    def eval_with_proof(self, key: VDPFKey, positions: List[int]) -> Tuple[List[int], VDPFProof]:
        """
        Evaluate VDPF at multiple positions and generate proof
        """
        results = []
        
        # Initialize pi = cs
        pi = [block[:] for block in key.cs]  # Deep copy
        
        # Evaluate each position
        for x in positions:
            # Standard DPF evaluation - VDPFKey inherits from DPFKey
            result = self.eval(key, x)
            results.append(result)
            
            # Get seed at position x
            seed = self._get_final_seed(key, x)
            bit = seed[0] & 1
            
            # Verification update
            x_bytes = x.to_bytes(16, 'big')
            
            # Step 1: tpi = H(x || seed)
            tpi = self.mmo.hash_2to4(x_bytes, seed)
            
            # Step 2: Compute pi ⊕ correct(tpi, cs, bit)
            hashinput = []
            for i in range(4):
                if bit == 0:
                    corrected = tpi[i]
                else:
                    corrected = bytes(a ^ b for a, b in zip(tpi[i], key.cs[i]))
                
                pi_corrected = bytes(a ^ b for a, b in zip(pi[i], corrected))
                hashinput.append(pi_corrected)
            
            # Step 3: cpi = H'(pi ⊕ correct(tpi, cs, bit))
            cpi = self.mmo.hash_4to4(hashinput)
            
            # Update pi
            for i in range(4):
                pi[i] = bytes(a ^ b for a, b in zip(pi[i], cpi[i]))
        
        # Generate final proof: SHA-256 of pi
        h = hashlib.sha256()
        for block in pi:
            h.update(block)
        proof = VDPFProof(h.digest())
        
        return results, proof
    
    def verify(self, proof0: VDPFProof, proof1: VDPFProof) -> bool:
        """
        Verify that two proofs are consistent
        """
        return proof0.hash_value == proof1.hash_value
    
    def _get_final_seed(self, key: DPFKey, x: int) -> bytes:
        """
        Helper to get the final seed when evaluating at position x
        """
        # Initialize
        seed = key.initial_seed
        bit = key.initial_bit
        
        # Traverse the tree
        for i in range(key.domain_bits):
            # PRG expansion
            sL, sR, tL, tR = self._dpf_prg(seed)
            
            # Get correction words
            sCW, tCW0, tCW1 = key.correction_words[i]
            
            # Apply corrections if control bit is 1
            if bit == 1:
                sL = self._xor_bytes(sL, sCW)
                sR = self._xor_bytes(sR, sCW)
                tL = tL ^ tCW0
                tR = tR ^ tCW1
            
            # Get x's bit at this level
            x_bit = self._get_bit(x, key.domain_bits, i + 1)
            
            # Move to next level
            if x_bit == 0:
                seed = sL
                bit = tL
            else:
                seed = sR
                bit = tR
        
        return seed


def test_verifiable_dpf():
    """Test the verifiable DPF implementation"""
    print("Testing Verifiable DPF Implementation")
    print("=" * 50)
    
    vdpf = VerifiableDPF()
    
    # Test parameters
    domain_bits = 4
    domain_size = 2**domain_bits
    alpha = 5
    
    print(f"\nTest Setup:")
    print(f"Domain: [0, {domain_size})")
    print(f"Target position α = {alpha}")
    
    # Generate VDPF keys
    print("\nGenerating VDPF keys...")
    vdpf_key0, vdpf_key1 = vdpf.gen(domain_bits, alpha)
    print("✓ Keys generated successfully")
    
    # Evaluate on entire domain
    positions = list(range(domain_size))
    
    print("\nEvaluating with proof generation...")
    results0, proof0 = vdpf.eval_with_proof(vdpf_key0, positions)
    results1, proof1 = vdpf.eval_with_proof(vdpf_key1, positions)
    
    # Check correctness
    print("\nVerification Results:")
    print("Position | Party 0 | Party 1 | XOR")
    print("-" * 40)
    
    for x in range(domain_size):
        xor_result = results0[x] ^ results1[x]
        expected = 1 if x == alpha else 0
        status = "✓" if xor_result == expected else "✗"
        print(f"{x:8d} | {results0[x]:7d} | {results1[x]:7d} | {xor_result:3d} {status}")
    
    # Verify proofs
    print(f"\nProof Verification:")
    print(f"Proof 0: {proof0.hash_value.hex()[:16]}...")
    print(f"Proof 1: {proof1.hash_value.hex()[:16]}...")
    
    is_valid = vdpf.verify(proof0, proof1)
    print(f"Verification: {'✓ PASS' if is_valid else '✗ FAIL'}")
    
    # Test with subset of positions
    print("\n\nTesting with subset evaluation:")
    subset = [3, 5, 7]
    _, proof0_subset = vdpf.eval_with_proof(vdpf_key0, subset)
    _, proof1_subset = vdpf.eval_with_proof(vdpf_key1, subset)
    
    print(f"Evaluated positions: {subset}")
    is_valid_subset = vdpf.verify(proof0_subset, proof1_subset)
    print(f"Subset verification: {'✓ PASS' if is_valid_subset else '✗ FAIL'}")


if __name__ == "__main__":
    test_verifiable_dpf()