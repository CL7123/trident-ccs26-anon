"""
Binary protocol communication module
Direct binary data transmission, avoiding JSON serialization overhead
"""
import struct
import socket
from typing import Dict, Any, Tuple


class BinaryProtocol:
    """Binary protocol handler class"""

    # Command type mapping
    CMD_QUERY_NODE_VECTOR = 1
    CMD_GET_STATUS = 2
    CMD_QUERY_NEIGHBOR_LIST = 3

    @staticmethod
    def encode_request(command: str, dpf_key: bytes = None, query_id: str = None) -> bytes:
        """Encode request to binary format"""
        # Protocol format:
        # [4 bytes: total length][1 byte: command type][4 bytes: query_id length][query_id][4 bytes: key length][key data]

        # Command mapping
        cmd_map = {
            'query_node_vector': BinaryProtocol.CMD_QUERY_NODE_VECTOR,
            'get_status': BinaryProtocol.CMD_GET_STATUS,
            'query_neighbor_list': BinaryProtocol.CMD_QUERY_NEIGHBOR_LIST
        }
        cmd_byte = cmd_map.get(command, 0)

        # Encode query_id
        query_id_bytes = query_id.encode('utf-8') if query_id else b''
        query_id_len = len(query_id_bytes)

        # Encode key
        key_len = len(dpf_key) if dpf_key else 0

        # Calculate total length (excluding the length field itself)
        total_len = 1 + 4 + query_id_len + 4 + key_len

        # Build binary data
        data = struct.pack('>I', total_len)  # Total length
        data += struct.pack('B', cmd_byte)   # Command type
        data += struct.pack('>I', query_id_len)  # query_id length
        data += query_id_bytes               # query_id data
        data += struct.pack('>I', key_len)   # Key length
        if dpf_key:
            data += dpf_key                  # Key data

        return data

    @staticmethod
    def decode_request(data: bytes) -> Dict[str, Any]:
        """Decode binary request"""
        offset = 0

        # Read command type
        cmd_byte = struct.unpack_from('B', data, offset)[0]
        offset += 1

        # Command mapping
        cmd_map = {
            BinaryProtocol.CMD_QUERY_NODE_VECTOR: 'query_node_vector',
            BinaryProtocol.CMD_GET_STATUS: 'get_status',
            BinaryProtocol.CMD_QUERY_NEIGHBOR_LIST: 'query_neighbor_list'
        }
        command = cmd_map.get(cmd_byte, 'unknown')

        # Read query_id
        query_id_len = struct.unpack_from('>I', data, offset)[0]
        offset += 4
        query_id = data[offset:offset+query_id_len].decode('utf-8') if query_id_len > 0 else None
        offset += query_id_len

        # Read key
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
        """Send binary request"""
        data = BinaryProtocol.encode_request(command, dpf_key, query_id)
        sock.sendall(data)

    @staticmethod
    def receive_binary_request(sock: socket.socket) -> Dict[str, Any]:
        """Receive binary request"""
        # First read 4 bytes of length
        length_data = sock.recv(4)
        if not length_data:
            return None

        total_len = struct.unpack('>I', length_data)[0]

        # Read remaining data
        data = b''
        while len(data) < total_len:
            chunk = sock.recv(min(4096, total_len - len(data)))
            if not chunk:
                break
            data += chunk

        return BinaryProtocol.decode_request(data)

    @staticmethod
    def encode_response(response_dict: Dict[str, Any]) -> bytes:
        """Encode response to binary format"""
        # Directly serialize entire response dictionary, maintaining original structure
        import json
        json_data = json.dumps(response_dict).encode('utf-8')
        # Add length prefix
        return struct.pack('>I', len(json_data)) + json_data

    @staticmethod
    def receive_response(sock: socket.socket) -> Dict[str, Any]:
        """Receive response"""
        # Read length
        length_data = sock.recv(4)
        if not length_data:
            return None

        total_len = struct.unpack('>I', length_data)[0]

        # Read JSON data
        data = b''
        while len(data) < total_len:
            chunk = sock.recv(min(4096, total_len - len(data)))
            if not chunk:
                break
            data += chunk

        import json
        return json.loads(data.decode('utf-8'))
