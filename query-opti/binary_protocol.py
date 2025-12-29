"""
二进制协议通信模块
直接传输二进制数据，避免JSON序列化开销
"""
import struct
import socket
from typing import Dict, Any, Tuple


class BinaryProtocol:
    """二进制协议处理类"""
    
    # 命令类型映射
    CMD_QUERY_NODE_VECTOR = 1
    CMD_GET_STATUS = 2
    CMD_QUERY_NEIGHBOR_LIST = 3
    
    @staticmethod
    def encode_request(command: str, dpf_key: bytes = None, query_id: str = None) -> bytes:
        """编码请求为二进制格式"""
        # 协议格式：
        # [4字节: 总长度][1字节: 命令类型][4字节: query_id长度][query_id][4字节: key长度][key数据]
        
        # 命令映射
        cmd_map = {
            'query_node_vector': BinaryProtocol.CMD_QUERY_NODE_VECTOR,
            'get_status': BinaryProtocol.CMD_GET_STATUS,
            'query_neighbor_list': BinaryProtocol.CMD_QUERY_NEIGHBOR_LIST
        }
        cmd_byte = cmd_map.get(command, 0)
        
        # 编码query_id
        query_id_bytes = query_id.encode('utf-8') if query_id else b''
        query_id_len = len(query_id_bytes)
        
        # 编码密钥
        key_len = len(dpf_key) if dpf_key else 0
        
        # 计算总长度（不包括长度字段本身）
        total_len = 1 + 4 + query_id_len + 4 + key_len
        
        # 构建二进制数据
        data = struct.pack('>I', total_len)  # 总长度
        data += struct.pack('B', cmd_byte)   # 命令类型
        data += struct.pack('>I', query_id_len)  # query_id长度
        data += query_id_bytes               # query_id数据
        data += struct.pack('>I', key_len)   # 密钥长度
        if dpf_key:
            data += dpf_key                  # 密钥数据
        
        return data
    
    @staticmethod
    def decode_request(data: bytes) -> Dict[str, Any]:
        """解码二进制请求"""
        offset = 0
        
        # 读取命令类型
        cmd_byte = struct.unpack_from('B', data, offset)[0]
        offset += 1
        
        # 命令映射
        cmd_map = {
            BinaryProtocol.CMD_QUERY_NODE_VECTOR: 'query_node_vector',
            BinaryProtocol.CMD_GET_STATUS: 'get_status',
            BinaryProtocol.CMD_QUERY_NEIGHBOR_LIST: 'query_neighbor_list'
        }
        command = cmd_map.get(cmd_byte, 'unknown')
        
        # 读取query_id
        query_id_len = struct.unpack_from('>I', data, offset)[0]
        offset += 4
        query_id = data[offset:offset+query_id_len].decode('utf-8') if query_id_len > 0 else None
        offset += query_id_len
        
        # 读取密钥
        key_len = struct.unpack_from('>I', data, offset)[0]
        offset += 4
        dpf_key = data[offset:offset+key_len] if key_len > 0 else None
        
        result = {'command': command}
        if query_id:
            result['query_id'] = query_id
        if dpf_key:
            result['dpf_key'] = dpf_key
            
        return result
    
    @staticmethod
    def send_binary_request(sock: socket.socket, command: str, dpf_key: bytes = None, query_id: str = None):
        """发送二进制请求"""
        data = BinaryProtocol.encode_request(command, dpf_key, query_id)
        sock.sendall(data)
    
    @staticmethod
    def receive_binary_request(sock: socket.socket) -> Dict[str, Any]:
        """接收二进制请求"""
        # 先读取4字节的长度
        length_data = sock.recv(4)
        if not length_data:
            return None
            
        total_len = struct.unpack('>I', length_data)[0]
        
        # 读取剩余数据
        data = b''
        while len(data) < total_len:
            chunk = sock.recv(min(4096, total_len - len(data)))
            if not chunk:
                break
            data += chunk
            
        return BinaryProtocol.decode_request(data)
    
    @staticmethod
    def encode_response(response_dict: Dict[str, Any]) -> bytes:
        """编码响应为二进制格式"""
        # 直接序列化整个响应字典，保持原有结构
        import json
        json_data = json.dumps(response_dict).encode('utf-8')
        # 添加长度前缀
        return struct.pack('>I', len(json_data)) + json_data
    
    @staticmethod
    def receive_response(sock: socket.socket) -> Dict[str, Any]:
        """接收响应"""
        # 读取长度
        length_data = sock.recv(4)
        if not length_data:
            return None
            
        total_len = struct.unpack('>I', length_data)[0]
        
        # 读取JSON数据
        data = b''
        while len(data) < total_len:
            chunk = sock.recv(min(4096, total_len - len(data)))
            if not chunk:
                break
            data += chunk
            
        import json
        return json.loads(data.decode('utf-8'))