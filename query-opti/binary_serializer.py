"""
高效的二进制密钥序列化器
替代pickle+base64，提升序列化/反序列化性能
"""
import struct
import numpy as np
from typing import List, Tuple, Any
from dataclasses import dataclass


class BinaryKeySerializer:
    """VDPF密钥的二进制序列化器"""
    
    @staticmethod
    def serialize_dpf_key(key) -> bytes:
        """序列化DPF密钥为二进制格式"""
        # 头部：party_id(1B) + domain_bits(1B) + initial_bit(1B) + seed_len(1B)
        seed_len = len(key.initial_seed)
        header = struct.pack('BBBB', key.party_id, key.domain_bits, key.initial_bit, seed_len)
        
        # 初始种子
        seed_data = key.initial_seed
        
        # 修正词数量
        num_cw = len(key.correction_words)
        cw_count = struct.pack('H', num_cw)  # 2字节表示修正词数量
        
        # 修正词数据
        cw_data = b''
        for sCW, tCW0, tCW1 in key.correction_words:
            cw_len = len(sCW)
            # 每个修正词：长度(1B) + sCW + tCW0(1B) + tCW1(1B)
            cw_data += struct.pack('B', cw_len) + sCW + struct.pack('BB', tCW0, tCW1)
        
        # 最后的修正词
        last_cw = struct.pack('I', key.last_cw)
        
        return header + seed_data + cw_count + cw_data + last_cw
    
    @staticmethod
    def deserialize_dpf_key(data: bytes, offset: int = 0):
        """从二进制数据反序列化DPF密钥"""
        # 解析头部
        party_id, domain_bits, initial_bit, seed_len = struct.unpack_from('BBBB', data, offset)
        offset += 4
        
        # 解析初始种子
        initial_seed = data[offset:offset+seed_len]
        offset += seed_len
        
        # 解析修正词数量
        num_cw, = struct.unpack_from('H', data, offset)
        offset += 2
        
        # 解析修正词
        correction_words = []
        for _ in range(num_cw):
            cw_len, = struct.unpack_from('B', data, offset)
            offset += 1
            sCW = data[offset:offset+cw_len]
            offset += cw_len
            tCW0, tCW1 = struct.unpack_from('BB', data, offset)
            offset += 2
            correction_words.append((sCW, tCW0, tCW1))
        
        # 解析最后的修正词
        last_cw, = struct.unpack_from('I', data, offset)
        offset += 4
        
        # 重建密钥对象
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
        """序列化VDPF密钥（包含DPF密钥和额外的cs数据）"""
        # 先序列化基础DPF密钥
        dpf_data = BinaryKeySerializer.serialize_dpf_key(key)
        
        # 序列化cs列表
        num_cs = len(key.cs)
        cs_header = struct.pack('B', num_cs)
        cs_data = b''.join(key.cs)
        
        return dpf_data + cs_header + cs_data
    
    @staticmethod
    def deserialize_vdpf_key(data: bytes, offset: int = 0):
        """反序列化VDPF密钥"""
        # 先反序列化DPF部分
        dpf_key, new_offset = BinaryKeySerializer.deserialize_dpf_key(data, offset)
        
        # 反序列化cs部分
        num_cs, = struct.unpack_from('B', data, new_offset)
        new_offset += 1
        
        cs = []
        cs_len = 16  # 假设每个cs是16字节
        for _ in range(num_cs):
            cs_item = data[new_offset:new_offset+cs_len]
            cs.append(cs_item)
            new_offset += cs_len
        
        # 重建VDPF密钥
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
        """序列化VDPF23密钥（包含两个VDPFPlus密钥）"""
        # 头部：party_id(1B)
        header = struct.pack('B', key.party_id)
        
        # 序列化g_key (VDPFPlus)
        g_z = struct.pack('I', key.g_key.z)
        g_vdpf_data = BinaryKeySerializer.serialize_vdpf_key(key.g_key.vdpf_key)
        
        # 序列化k_key (VDPFPlus)
        k_z = struct.pack('I', key.k_key.z)
        k_vdpf_data = BinaryKeySerializer.serialize_vdpf_key(key.k_key.vdpf_key)
        
        return header + g_z + g_vdpf_data + k_z + k_vdpf_data
    
    @staticmethod
    def deserialize_vdpf23_key(data: bytes) -> Any:
        """反序列化VDPF23密钥"""
        offset = 0
        
        # 解析party_id
        party_id, = struct.unpack_from('B', data, offset)
        offset += 1
        
        # 解析g_key
        g_z, = struct.unpack_from('I', data, offset)
        offset += 4
        g_vdpf_key, offset = BinaryKeySerializer.deserialize_vdpf_key(data, offset)
        
        # 解析k_key
        k_z, = struct.unpack_from('I', data, offset)
        offset += 4
        k_vdpf_key, offset = BinaryKeySerializer.deserialize_vdpf_key(data, offset)
        
        # 重建密钥对象
        from standardDPF.vdpf_plus import VDPFPlusKey
        from standardDPF.vdpf_23 import VDPF23Key
        
        g_key = VDPFPlusKey(z=g_z, vdpf_key=g_vdpf_key)
        k_key = VDPFPlusKey(z=k_z, vdpf_key=k_vdpf_key)
        
        return VDPF23Key(party_id=party_id, g_key=g_key, k_key=k_key)

