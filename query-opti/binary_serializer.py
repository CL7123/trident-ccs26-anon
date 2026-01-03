"""
Efficient binary key serializer
Replaces pickle+base64, improves serialization/deserialization performance
"""
import struct
import numpy as np
from typing import List, Tuple, Any
from dataclasses import dataclass


class BinaryKeySerializer:
    """Binary serializer for VDPF keys"""
    
    @staticmethod
    def serialize_dpf_key(key) -> bytes:
        """Serialize DPF key to binary format"""
        # Header: party_id(1B) + domain_bits(1B) + initial_bit(1B) + seed_len(1B)
        seed_len = len(key.initial_seed)
        header = struct.pack('BBBB', key.party_id, key.domain_bits, key.initial_bit, seed_len)

        # Initial seed
        seed_data = key.initial_seed

        # Correction word count
        num_cw = len(key.correction_words)
        cw_count = struct.pack('H', num_cw)  # 2 bytes to represent correction word count

        # Correction word data
        cw_data = b''
        for sCW, tCW0, tCW1 in key.correction_words:
            cw_len = len(sCW)
            # Each correction word: length(1B) + sCW + tCW0(1B) + tCW1(1B)
            cw_data += struct.pack('B', cw_len) + sCW + struct.pack('BB', tCW0, tCW1)

        # Last correction word
        last_cw = struct.pack('I', key.last_cw)

        return header + seed_data + cw_count + cw_data + last_cw
    
    @staticmethod
    def deserialize_dpf_key(data: bytes, offset: int = 0):
        """Deserialize DPF key from binary data"""
        # Parse header
        party_id, domain_bits, initial_bit, seed_len = struct.unpack_from('BBBB', data, offset)
        offset += 4

        # Parse initial seed
        initial_seed = data[offset:offset+seed_len]
        offset += seed_len

        # Parse correction word count
        num_cw, = struct.unpack_from('H', data, offset)
        offset += 2

        # Parse correction words
        correction_words = []
        for _ in range(num_cw):
            cw_len, = struct.unpack_from('B', data, offset)
            offset += 1
            sCW = data[offset:offset+cw_len]
            offset += cw_len
            tCW0, tCW1 = struct.unpack_from('BB', data, offset)
            offset += 2
            correction_words.append((sCW, tCW0, tCW1))

        # Parse last correction word
        last_cw, = struct.unpack_from('I', data, offset)
        offset += 4

        # Reconstruct key object
        from standardDPF.standard_dpf import DPFKey
        key = DPFKey(
            party_id=party_id,
            initial_seed=initial_seed,
            initial_bit=initial_bit,
            correction_words=correction_words,
            last_cw=last_cw,
            domain_bits=domain_bits
        )

        return key, offset
    
    @staticmethod
    def serialize_vdpf_key(key) -> bytes:
        """Serialize VDPF key (contains DPF key and additional cs data)"""
        # First serialize base DPF key
        dpf_data = BinaryKeySerializer.serialize_dpf_key(key)

        # Serialize cs list
        num_cs = len(key.cs)
        cs_header = struct.pack('B', num_cs)
        cs_data = b''.join(key.cs)

        return dpf_data + cs_header + cs_data
    
    @staticmethod
    def deserialize_vdpf_key(data: bytes, offset: int = 0):
        """Deserialize VDPF key"""
        # First deserialize DPF part
        dpf_key, new_offset = BinaryKeySerializer.deserialize_dpf_key(data, offset)

        # Deserialize cs part
        num_cs, = struct.unpack_from('B', data, new_offset)
        new_offset += 1

        cs = []
        cs_len = 16  # Assume each cs is 16 bytes
        for _ in range(num_cs):
            cs_item = data[new_offset:new_offset+cs_len]
            cs.append(cs_item)
            new_offset += cs_len

        # Reconstruct VDPF key
        from standardDPF.verifiable_dpf import VDPFKey
        vdpf_key = VDPFKey(
            party_id=dpf_key.party_id,
            initial_seed=dpf_key.initial_seed,
            initial_bit=dpf_key.initial_bit,
            correction_words=dpf_key.correction_words,
            last_cw=dpf_key.last_cw,
            domain_bits=dpf_key.domain_bits,
            cs=cs
        )

        return vdpf_key, new_offset
    
    @staticmethod
    def serialize_vdpf23_key(key) -> bytes:
        """Serialize VDPF23 key (contains two VDPFPlus keys)"""
        # Header: party_id(1B)
        header = struct.pack('B', key.party_id)

        # Serialize g_key (VDPFPlus)
        g_z = struct.pack('I', key.g_key.z)
        g_vdpf_data = BinaryKeySerializer.serialize_vdpf_key(key.g_key.vdpf_key)

        # Serialize k_key (VDPFPlus)
        k_z = struct.pack('I', key.k_key.z)
        k_vdpf_data = BinaryKeySerializer.serialize_vdpf_key(key.k_key.vdpf_key)

        return header + g_z + g_vdpf_data + k_z + k_vdpf_data
    
    @staticmethod
    def deserialize_vdpf23_key(data: bytes) -> Any:
        """Deserialize VDPF23 key"""
        offset = 0

        # Parse party_id
        party_id, = struct.unpack_from('B', data, offset)
        offset += 1

        # Parse g_key
        g_z, = struct.unpack_from('I', data, offset)
        offset += 4
        g_vdpf_key, offset = BinaryKeySerializer.deserialize_vdpf_key(data, offset)

        # Parse k_key
        k_z, = struct.unpack_from('I', data, offset)
        offset += 4
        k_vdpf_key, offset = BinaryKeySerializer.deserialize_vdpf_key(data, offset)

        # Reconstruct key object
        from standardDPF.vdpf_plus import VDPFPlusKey
        from standardDPF.vdpf_23 import VDPF23Key

        g_key = VDPFPlusKey(z=g_z, vdpf_key=g_vdpf_key)
        k_key = VDPFPlusKey(z=k_z, vdpf_key=k_vdpf_key)

        return VDPF23Key(party_id=party_id, g_key=g_key, k_key=k_key)

