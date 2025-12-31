#!/usr/bin/env python3
"""
(2,3)-VDPF Implementation
Based on the construction in 23vdpf.md
"""

import secrets
from typing import Tuple, List, Dict
from dataclasses import dataclass
import sys
import os

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
# Import MPC functionality for (2,3)-SSS
sys.path.append('~/trident/src')

from vdpf_plus import VDPFPlus, VDPFPlusKey
from basic_functionalities import MPC23SSS, Share


@dataclass
class VDPF23Key:
    """(2,3)-VDPF key structure"""
    party_id: int  # 1, 2, or 3
    g_key: VDPFPlusKey  # First VDPF+ key (g0 or g1)
    k_key: VDPFPlusKey  # Second VDPF+ key (k0 or k1)
    

class VDPF23:
    """
    (2,3)-VDPF implementation
    Constructs 3-party VDPF from two (2,2)-VDPF instances
    """
    
    def __init__(self, domain_bits: int, security_param: int = 128, field_size: int = None, dataset_name: str = None):
        self.domain_bits = domain_bits
        self.security_param = security_param
        self.l = security_param  # Binary field parameter F_2^l
        
        # Set field size: should be a prime slightly larger than 2^l
        if field_size is None:
            if security_param == 31:
                # For l=31, use the next prime after 2^31
                self.field_size = 2147483659  # This is prime and > 2^31
            elif security_param == 16:
                # For l=16, use the next prime after 2^16
                self.field_size = 65537  # This is prime and > 2^16
            elif security_param == 8:
                # For l=8, use the next prime after 2^8
                self.field_size = 257  # This is prime and > 2^8
            else:
                # Default to 2^31-1 for other cases
                self.field_size = 2**31 - 1
        else:
            self.field_size = field_size
            
        self.domain_size = 2 ** domain_bits
        
        # Initialize underlying (2,2)-VDPF+
        self.vdpf_plus_g = VDPFPlus(domain_bits)
        self.vdpf_plus_k = VDPFPlus(domain_bits)
        
        # Initialize (2,3)-SSS for secret sharing with correct field size
        if dataset_name:
            # [CN]Dataset[CN]
            from domain_config import get_config
            config = get_config(dataset_name)
            # [CN]
            config.domain_bits = self.domain_bits
            config.domain_size = self.domain_size
            self.sss = MPC23SSS(config)
        else:
            # [CN]createMPCinstance（[CN]）
            self.sss = MPC23SSS()
        
    def embed_prime_to_binary(self, x_prime: int) -> int:
        # Ensure x is within binary field range
        return x_prime % (2 ** self.l)
        
    def embed_binary_to_prime(self, x_binary: int) -> int:
        return x_binary % self.field_size
        
    def op_embed_xor(self, x_prime: int, y_binary: int) -> int:
        """
        ⊞ operation: embed prime field element to binary and XOR
        """
        x_binary = self.embed_prime_to_binary(x_prime)
        return x_binary ^ y_binary
        
    def op_embed_mult(self, x_prime: int, y_binary: int) -> int:
        """
        ⊙ operation: embed binary field element to prime and multiply
        """
        y_prime = self.embed_binary_to_prime(y_binary)
        return (x_prime * y_prime) % self.field_size
        
    def gen(self, alpha: int, beta: int) -> Tuple[VDPF23Key, VDPF23Key, VDPF23Key]:
        """
        Generate (2,3)-VDPF keys
        
        Args:
            alpha: Index position (where to place the value)
            beta: Value to place at position alpha
            
        Returns:
            Three VDPF23Key objects for parties 1, 2, and 3
        """
        # Step 1: (β₁, β₂, β₃) ← SS₂,₃.Share(β)
        beta_shares = self.sss.share_secret(beta)
        beta1 = beta_shares[0].value  # In prime field F
        beta2 = beta_shares[1].value  # In prime field F
        beta3 = beta_shares[2].value  # In prime field F
        
        # Step 2: β'ᵢ ← βᵢ · (i)⁻¹, for i ∈ {1, 2, 3}
        # Compute modular inverse of i in the prime field
        beta1_prime = (beta1 * pow(1, self.field_size - 2, self.field_size)) % self.field_size
        beta2_prime = (beta2 * pow(2, self.field_size - 2, self.field_size)) % self.field_size
        beta3_prime = (beta3 * pow(3, self.field_size - 2, self.field_size)) % self.field_size
        
        # Step 3: v₀ ← F_2^l (random value in binary field)
        # v₀ is chosen randomly in F_2^l
        v0 = secrets.randbelow(2**self.l)  # Random in F_2^l
        
        # Step 4-6: Compute v values using ⊞ operation
        # v₂ ← β'₁ ⊞ v₀
        v2 = self.op_embed_xor(beta1_prime, v0)
        
        # v₁ ← β'₂ ⊞ v₂
        v1 = self.op_embed_xor(beta2_prime, v2)
        
        # v₃ ← β'₃ ⊞ v₁
        v3 = self.op_embed_xor(beta3_prime, v1)
        
        # Step 7: (g₀, g₁) ← VDPF⁺.Gen(1^κ, α, v₀, v₁)
        # Generate VDPF+ keys that output v0 and v1 at position alpha
        g0, g1 = self.vdpf_plus_g.gen(alpha, v0, v1)
        
        # Step 8: (k₀, k₁) ← VDPF⁺.Gen(1^κ, α, v₂, v₃)
        # Generate VDPF+ keys that output v2 and v3 at position alpha
        k0, k1 = self.vdpf_plus_k.gen(alpha, v2, v3)
        
        # Step 9: Set f₁ = (g₀, k₀), f₂ = (g₁, k₀) and f₃ = (g₁, k₁)
        f1 = VDPF23Key(party_id=1, g_key=g0, k_key=k0)
        f2 = VDPF23Key(party_id=2, g_key=g1, k_key=k0)
        f3 = VDPF23Key(party_id=3, g_key=g1, k_key=k1)
        
        # Step 10: Output (f₁, f₂, f₃)
        return f1, f2, f3
        
        
    def eval(self, b: int, fb: VDPF23Key, x: int) -> int:
        """
        Evaluate (2,3)-VDPF at position x
        
        Args:
            b: Party ID (1, 2, or 3)
            fb: The key for party b
            x: Position to evaluate
            
        Returns:
            The output value at position x
        """
        # Step 1: Parse (g, k) ← fb
        g = fb.g_key
        k = fb.k_key
        
        # Step 2: Set (bᵍ, bₖ) based on party ID
        if b == 1:
            b_g, b_k = 0, 0
        elif b == 2:
            b_g, b_k = 1, 0
        else:  # b == 3
            b_g, b_k = 1, 1
            
        # Step 3-4: Compute yᵍ and yₖ using VDPF+.Eval
        y_g = self.vdpf_plus_g.eval(b_g, g, x)
        y_k = self.vdpf_plus_k.eval(b_k, k, x)
        
        # Step 5: Compute y = (yᵍ ⊕ yₖ) ⊙ b
        # The ⊙ operation embeds to prime field and multiplies
        y_xor = y_g ^ y_k
        y = self.op_embed_mult(b, y_xor)
        
        return y
        
    def prove(self, b: int, fb: VDPF23Key, rb: int, positions: List[int]) -> Dict:
        """
        Generate proof for (2,3)-VDPF
        
        Args:
            b: Party ID (1, 2, or 3)
            fb: The key for party b
            rb: Random value for party b
            positions: List of positions to evaluate
            
        Returns:
            Proof dictionary containing all proof components
        """
        # Step 1: Parse (g, k) ← fb
        g = fb.g_key
        k = fb.k_key
        
        # Step 2: Set (bᵍ, bₖ) based on party ID
        if b == 1:
            b_g, b_k = 0, 0
        elif b == 2:
            b_g, b_k = 1, 0
        else:  # b == 3
            b_g, b_k = 1, 1
            
        # Step 3: πᵍ = VDPF⁺.Prove(bᵍ, g) and πₖ = VDPF⁺.Prove(bₖ, k)
        pi_g, z_g = self.vdpf_plus_g.prove(b_g, g, positions)
        pi_k, z_k = self.vdpf_plus_k.prove(b_k, k, positions)
        
        # Step 4: Initialize yb = {}, u = {}
        yb = []
        u = []
        
        # Step 5: For each position
        for xi in positions:
            # (a) Compute yᵍ(xᵢ) = VDPF⁺.Eval(bᵍ, g, xᵢ)
            y_g_xi = self.vdpf_plus_g.eval(b_g, g, xi)
            
            # (b) Compute yₖ(xᵢ) = VDPF⁺.Eval(bₖ, k, xᵢ)
            y_k_xi = self.vdpf_plus_k.eval(b_k, k, xi)
            
            # (c) yᵢ ← (yᵍ(xᵢ) ⊕ yₖ(xᵢ)) ⊙ b
            y_xor = y_g_xi ^ y_k_xi
            yi = self.op_embed_mult(b, y_xor)
            yb.append(yi)
            
            # (e) Generate the same uᵢ ← F and let u ← u ∪ {uᵢ}
            # For simplicity, use deterministic generation based on position
            ui = (xi * 7919 + 4933) % self.field_size  # Simple deterministic generation
            u.append(ui)
            
        # Step 6: βb ← ∑ⁿᵢ₌₁ yb[i]
        beta_b = sum(yb) % self.field_size
        
        # Step 7: tb ← (yb · u)² - βb(yb · u²) - rb (over F)
        # Compute yb · u (dot product)
        yb_dot_u = sum(yb[i] * u[i] for i in range(len(yb))) % self.field_size
        
        # Compute yb · u²
        u_squared = [(ui * ui) % self.field_size for ui in u]
        yb_dot_u2 = sum(yb[i] * u_squared[i] for i in range(len(yb))) % self.field_size
        
        # Compute tb
        tb = ((yb_dot_u * yb_dot_u) - (beta_b * yb_dot_u2) - rb) % self.field_size
        
        # Step 8: Output πb = (πᵍ, πₖ, tb, H(tb))
        import hashlib
        h_tb = hashlib.sha256(str(tb).encode()).hexdigest()
        
        proof = {
            'pi_g': pi_g,
            'pi_k': pi_k,
            'tb': tb,
            'h_tb': h_tb,
            'z_g': z_g,
            'z_k': z_k
        }
        
        return proof
        
    def verify(self, proof1: Dict, proof2: Dict, proof3: Dict) -> bool:
        """
        Verify (2,3)-VDPF proofs
        
        Args:
            proof1, proof2, proof3: Proofs from parties 1, 2, and 3
            
        Returns:
            True if verification passes
        """
        # Step 1: Parse πb = (πᵍ,b, πₖ,b, tb, hb)
        # Extract components from each proof
        
        # Step 2: Compute t = ss.Reconstruct(t₁, t₂, t₃)
        t1 = proof1['tb']
        t2 = proof2['tb']
        t3 = proof3['tb']
        
        # Reconstruct using (2,3)-SSS
        shares = [
            Share(value=t1, party_id=1, degree=2),
            Share(value=t2, party_id=2, degree=2),
            Share(value=t3, party_id=3, degree=2)
        ]
        t = self.sss.reconstruct(shares)
        
        # Step 3: Verify conditions
        # Check πᵍ,₂ = πᵍ,₃
        if proof2['pi_g'].hash_value != proof3['pi_g'].hash_value:
            return False
            
        # Check πₖ,₁ = πₖ,₂
        if proof1['pi_k'].hash_value != proof2['pi_k'].hash_value:
            return False
            
        # Check VDPF⁺.Verify(πᵍ,₁, πᵍ,₂)
        if not self.vdpf_plus_g.verify((proof1['pi_g'], proof1['z_g']), 
                                       (proof2['pi_g'], proof2['z_g'])):
            return False
            
        # Check VDPF⁺.Verify(πₖ,₁, πₖ,₃)
        if not self.vdpf_plus_k.verify((proof1['pi_k'], proof1['z_k']), 
                                       (proof3['pi_k'], proof3['z_k'])):
            return False
            
        # Check t = 0
        if t != 0:
            return False
            
        # Check H(tb) = hb for all parties
        for i, proof in enumerate([proof1, proof2, proof3], 1):
            import hashlib
            expected_hash = hashlib.sha256(str(proof['tb']).encode()).hexdigest()
            if proof['h_tb'] != expected_hash:
                return False
                
        return True