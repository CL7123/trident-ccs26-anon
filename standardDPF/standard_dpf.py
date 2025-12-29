#!/usr/bin/env python3


import secrets
import hashlib
import struct
from typing import Tuple, List, Dict
from dataclasses import dataclass


@dataclass
class DPFKey:
    """DPF key structure"""
    party_id: int  # 0 or 1
    initial_seed: bytes
    initial_bit: int
    correction_words: List[Tuple[bytes, int, int]]  # [(sCW, tCW0, tCW1)]
    last_cw: int  # Final correction word (as integer for simplicity)
    domain_bits: int

class StandardDPF:
    """
    Standard (2,2)-DPF implementation following the C code logic
    """
    
    def __init__(self, security_param: int = 128):
        self.lambda_sec = security_param
        self.seed_len = security_param // 8 
        
    def gen(self, domain_bits: int, index: int, beta: int = 1) -> Tuple[DPFKey, DPFKey]:
        """
        Generate DPF keys following the C implementation logic
        
        Args:
            domain_bits: Number of bits in the domain
            index: Target index where output should be non-zero
            beta: The value that the two parties' outputs should XOR to at the target index (default: 1)
        """
        # Initialize seeds and bits
        seeds0 = [secrets.token_bytes(self.seed_len)]
        seeds1 = [secrets.token_bytes(self.seed_len)]
        bits0 = [0]
        bits1 = [1]
        
        # Correction words storage
        correction_words = []
        
        # Main loop - build the GGM tree
        for i in range(1, domain_bits + 1):
            # PRG expansion
            s0_L, s0_R, t0_L, t0_R = self._dpf_prg(seeds0[i-1])
            s1_L, s1_R, t1_L, t1_R = self._dpf_prg(seeds1[i-1])
            
            # Get index bit at this level
            index_bit = self._get_bit(index, domain_bits, i)
            
            # Determine keep/lose directions
            if index_bit == 0:
                keep = 0  # LEFT
                lose = 1  # RIGHT
                s0_keep, s0_lose = s0_L, s0_R
                s1_keep, s1_lose = s1_L, s1_R
                t0_keep = t0_L
                t1_keep = t1_L
            else:
                keep = 1  # RIGHT
                lose = 0  # LEFT
                s0_keep, s0_lose = s0_R, s0_L
                s1_keep, s1_lose = s1_R, s1_L
                t0_keep = t0_R
                t1_keep = t1_R
            
            # Compute correction words (following C code exactly)
            sCW = self._xor_bytes(s0_lose, s1_lose)
            tCW0 = t0_L ^ t1_L ^ index_bit ^ 1
            tCW1 = t0_R ^ t1_R ^ index_bit
            
            correction_words.append((sCW, tCW0, tCW1))
            
            # Update party 0 state
            if bits0[i-1] == 1:
                seeds0.append(self._xor_bytes(s0_keep, sCW))
                if keep == 0:  # LEFT
                    bits0.append(t0_keep ^ tCW0)
                else:  # RIGHT
                    bits0.append(t0_keep ^ tCW1)
            else:
                seeds0.append(s0_keep)
                bits0.append(t0_keep)
                
            # Update party 1 state
            if bits1[i-1] == 1:
                seeds1.append(self._xor_bytes(s1_keep, sCW))
                if keep == 0:  # LEFT
                    bits1.append(t1_keep ^ tCW0)
                else:  # RIGHT
                    bits1.append(t1_keep ^ tCW1)
            else:
                seeds1.append(s1_keep)
                bits1.append(t1_keep)
        
        # Compute final correction word
        # Convert final seeds to integers for XOR
        sFinal0 = int.from_bytes(seeds0[domain_bits][:4], 'big')
        sFinal1 = int.from_bytes(seeds1[domain_bits][:4], 'big')
        lastCW = beta ^ sFinal0 ^ sFinal1
        
        # Create keys
        key0 = DPFKey(
            party_id=0,
            initial_seed=seeds0[0],
            initial_bit=0,
            correction_words=correction_words,
            last_cw=lastCW,
            domain_bits=domain_bits
        )
        
        key1 = DPFKey(
            party_id=1,
            initial_seed=seeds1[0],
            initial_bit=1,
            correction_words=correction_words,
            last_cw=lastCW,
            domain_bits=domain_bits
        )
        
        
        return key0, key1
    
    def eval(self, key: DPFKey, x: int) -> int:
        """
        Evaluate DPF at position x
        Returns an integer for simplicity (representing the output)
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
        
        # Convert final seed to output
        res = int.from_bytes(seed[:4], 'big')
        
        # Apply final correction if control bit is 1
        if bit == 1:
            res = res ^ key.last_cw
            
        return res
    
    def _dpf_prg(self, seed: bytes) -> Tuple[bytes, bytes, int, int]:
        """PRG expansion using SHA-256"""
        # Left expansion
        h_left = hashlib.sha256()
        h_left.update(seed)
        h_left.update(b'\x00')
        left_output = h_left.digest()
        
        # Right expansion
        h_right = hashlib.sha256()
        h_right.update(seed)
        h_right.update(b'\x01')
        right_output = h_right.digest()
        
        # Extract seeds and control bits
        s_left = left_output[:self.seed_len]
        s_right = right_output[:self.seed_len]
        t_left = left_output[self.seed_len] & 1
        t_right = right_output[self.seed_len] & 1
        
        return s_left, s_right, t_left, t_right
    
    def _get_bit(self, value: int, total_bits: int, position: int) -> int:
        """Get bit at position (1-indexed from MSB)"""
        return (value >> (total_bits - position)) & 1
    
    def _xor_bytes(self, a: bytes, b: bytes) -> bytes:
        """XOR two byte arrays"""
        return bytes(x ^ y for x, y in zip(a, b))


