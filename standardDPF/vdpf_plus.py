#!/usr/bin/env python3
"""
VDPF+ Implementation
Extension of VDPF that supports custom output values at target position
"""

from typing import Tuple
from dataclasses import dataclass
from verifiable_dpf import VerifiableDPF, VDPFKey, VDPFProof


@dataclass
class VDPFPlusKey:
    """VDPF+ key structure"""
    z: int  # Adjustment value
    vdpf_key: VDPFKey  # Underlying VDPF key
    

class VDPFPlus:
    """
    VDPF+ implementation that supports custom output values
    """
    
    def __init__(self, domain_bits: int):
        self.domain_bits = domain_bits
        self.vdpf = VerifiableDPF()
        
    def gen(self, alpha: int, beta0: int, beta1: int) -> Tuple[VDPFPlusKey, VDPFPlusKey]:
        """
        Generate VDPF+ keys that output beta0 for party 0 and beta1 for party 1 at position alpha
        
        Args:
            alpha: Target position
            beta0: Output value for party 0 at position alpha
            beta1: Output value for party 1 at position alpha
            
        Returns:
            Two VDPF+ keys
        """
        # Step 1: β ← β₀ ⊕ β₁
        beta = beta0 ^ beta1
        
        # Step 2: (f₀, f₁) ← VDPF.Gen(1^κ, α, β)
        # Generate a modified VDPF that outputs beta at position alpha
        # We need to create a custom VDPF that outputs beta instead of 1
        
        # For now, we'll generate keys that output shares of beta
        # In a proper implementation, we'd modify the VDPF generation
        # to embed beta into the correction words
        
        # Generate VDPF keys that output beta at position alpha
        f0_std, f1_std = self.vdpf.gen(self.domain_bits, alpha, beta)
        
        # Evaluate to get what standard VDPF outputs at alpha
        eval0_std = self.vdpf.eval(f0_std, alpha)
        eval1_std = self.vdpf.eval(f1_std, alpha)
        
        # Standard VDPF ensures eval0_std ^ eval1_std = 1 at position alpha
        # We need to scale this to beta
        
        # Create scaled VDPF keys by modifying the last correction word
        # This is a simplified approach - proper implementation would modify
        # the entire tree structure
        
        # For binary field operations:
        # If standard VDPF gives shares s0, s1 where s0 ^ s1 = 1
        # We want shares beta0, beta1 where beta0 ^ beta1 = beta
        
        # Step 3: z ← β₀ ⊕ (scaled evaluation at α)
        # The offset z ensures party 0 gets beta0 and party 1 gets beta1
        
        # Since we need beta0 ^ beta1 = beta, and standard VDPF gives us
        # values that XOR to 1, we need to handle this differently
        
        # Temporary fix: use the values directly as correction
        # In practice, we'd need to modify the VDPF tree structure
        
        # Verify that VDPF outputs correctly at alpha
        if eval0_std ^ eval1_std != beta:
            raise ValueError(f"VDPF failed: {eval0_std} ^ {eval1_std} = {eval0_std ^ eval1_std}, expected {beta}")
            
        # Now the VDPF outputs beta at position alpha
        
        # The key insight: we need to transform VDPF outputs
        # Standard VDPF: outputs shares that XOR to 1
        # We need: outputs that XOR to beta
        
        # If standard VDPF outputs (s0, s1) where s0 ^ s1 = 1
        # We can scale by XORing with appropriate values
        
        # Step 3: z ← β₀ ⊕ VDPF.Eval(0, f₀, α)
        # This ensures party 0 outputs beta0 at position alpha
        z = beta0 ^ eval0_std
        
        # Step 4: f_b ← (z, f_b) for b ∈ {0, 1}
        key0 = VDPFPlusKey(z=z, vdpf_key=f0_std)
        key1 = VDPFPlusKey(z=z, vdpf_key=f1_std)
        
        return key0, key1
        
    def eval(self, b: int, fb: VDPFPlusKey, x: int) -> int:
        """
        Evaluate VDPF+ at position x
        
        Args:
            b: Party ID (0 or 1)
            fb: VDPF+ key for party b
            x: Position to evaluate
            
        Returns:
            Output value at position x
        """
        # Step 1: (z, f'_b) ← f_b
        z = fb.z
        f_prime_b = fb.vdpf_key
        
        # Step 2: y ← VDPF.Eval(b, f'_b, x) ⊕ z
        y_prime = self.vdpf.eval(f_prime_b, x)
        y = y_prime ^ z
        
        return y
        
    def prove(self, b: int, fb: VDPFPlusKey, positions: list) -> Tuple[VDPFProof, int]:
        """
        Generate proof for VDPF+
        
        Args:
            b: Party ID (0 or 1)
            fb: VDPF+ key for party b
            positions: List of positions to evaluate
            
        Returns:
            Proof and adjustment value z
        """
        # Step 1: (z, f'_b) ← f_b
        z = fb.z
        f_prime_b = fb.vdpf_key
        
        # Step 2: π'_b ← VDPF.Prove(b, f'_b)
        _, proof = self.vdpf.eval_with_proof(f_prime_b, positions)
        
        # Step 3: Output π_b = (π'_b, z)
        return proof, z
        
    def verify(self, proof0: Tuple[VDPFProof, int], proof1: Tuple[VDPFProof, int]) -> bool:
        """
        Verify VDPF+ proofs
        
        Args:
            proof0: Proof from party 0 (VDPFProof, z)
            proof1: Proof from party 1 (VDPFProof, z)
            
        Returns:
            True if verification passes
        """
        vdpf_proof0, z0 = proof0
        vdpf_proof1, z1 = proof1
        
        # Check that z values match
        if z0 != z1:
            return False
            
        # Verify underlying VDPF proofs
        return self.vdpf.verify(vdpf_proof0, vdpf_proof1)