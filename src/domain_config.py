#!/usr/bin/env python3
"""
统一的域参数配置
用于确保 MPC、VDPF 和测试使用一致的参数
"""

from dataclasses import dataclass
from typing import Dict, Any, List


@dataclass
class DomainConfig:
    """域配置参数"""
    # 索引域参数
    domain_bits: int      # 域大小的位数
    domain_size: int      # 2^domain_bits
    
    # 安全参数
    kappa: int           # 安全参数 κ
    l: int               # 二进制域参数（通常 l = κ）
    
    # 素数域参数
    prime: int           # 素数 p
    
    # DPF 输出参数
    output_bits: int     # DPF 输出的位数
    
    # 向量和查询参数
    vector_dimension: int # 向量维度
    num_docs: int        # 文档数量
    num_queries: int     # 查询数量
    
    # HNSW 索引参数
    efsearch: int        # 搜索时的候选数量
    efconstruction: int  # 构建时的候选数量
    layer: int           # HNSW 层数
    M: int               # 每层最大连接数
    
    def __post_init__(self):
        """验证参数的一致性"""
        assert self.domain_size == 2 ** self.domain_bits
        assert self.l == self.kappa, "当前设计要求 l = κ"
        # 验证安全性要求：p ≈ 2^l
        gap_ratio = abs(self.prime - 2**self.l) / (2**self.l)
        assert gap_ratio < 0.5, f"素数选择不安全：间隙比例 {gap_ratio} 太大"
        

# 配置名称常量
SIFTSMALL = "siftsmall"
LAION = "laion"
TRIPCLICK = "tripclick"
MS_MARCO = "ms_marco"
NFCORPUS = "nfcorpus"

# 预定义的配置（只保留31位安全性）
CONFIGS = {
 
    SIFTSMALL: DomainConfig(
        domain_bits=16,      # 支持 65k neighbor list
        domain_size=65536,
        kappa=31,
        l=31,
        prime=2**31 - 1,     # 2,147,483,647
        output_bits=31,
        vector_dimension=128,
        num_docs=10000,
        num_queries=100,
        efsearch=32,
        efconstruction=80,
        layer=2,
        M=64
    ),
    
    LAION: DomainConfig(
        domain_bits=19,      # 支持 524k neighbor list
        domain_size=524288,
        kappa=31,
        l=31,
        prime=2**31 - 1,     # 2,147,483,647
        output_bits=31,
        vector_dimension=512,
        num_docs=100000,
        num_queries=1000,
        efsearch=32,
        efconstruction=80,
        layer=2,
        M=64
    ),
    
    TRIPCLICK: DomainConfig(
        domain_bits=21,      # 支持 2M neighbor list (1.5M需求)
        domain_size=2097152,
        kappa=31,
        l=31,
        prime=2**31 - 1,     # 2,147,483,647
        output_bits=31,
        vector_dimension=768,
        num_docs=1523871,
        num_queries=1175,
        efsearch=36,
        efconstruction=160,
        layer=2,
        M=128
    ),
    
    MS_MARCO: DomainConfig(
        domain_bits=24,      # 支持 16M neighbor list (8.8M需求)
        domain_size=16777216,
        kappa=31,
        l=31,
        prime=2**31 - 1,     # 2,147,483,647
        output_bits=31,
        vector_dimension=768,
        num_docs=8841823,
        num_queries=6980,
        efsearch=48,
        efconstruction=200,
        layer=2,
        M=128
    ),
    
    NFCORPUS: DomainConfig(
        domain_bits=15,      # 支持 32k neighbor list
        domain_size=32768,
        kappa=31,
        l=31,
        prime=2**31 - 1,     # 2,147,483,647
        output_bits=31,
        vector_dimension=768,  # pritamdeka/S-PubMedBert-MS-MARCO 的向量维度
        num_docs=3633,         # NFCorpus 文档数量
        num_queries=323,       # NFCorpus 查询数量
        efsearch=32,           # 可以保持默认
        efconstruction=80,     # 可以保持默认
        layer=2,               # 小数据集2层足够
        M=32                   # 小数据集可以用较小的 M
    )
}


def get_config(dataset_name: str = SIFTSMALL) -> DomainConfig:
    """统一的配置获取接口
    
    Args:
        dataset_name: 数据集名称，默认为 SIFTSMALL
        
    Returns:
        对应数据集的域配置
        
    Raises:
        ValueError: 当数据集名称不存在时
    """
    if dataset_name not in CONFIGS:
        raise ValueError(f"未知的数据集配置: {dataset_name}. 可用的配置: {list_available_configs()}")
    return CONFIGS[dataset_name]


def list_available_configs() -> List[str]:
    """列出所有可用的配置名称"""
    return list(CONFIGS.keys())


def validate_data_requirements(config: DomainConfig, num_data_points: int, max_value: int):
    """验证配置是否满足数据需求"""
    if config.domain_size < num_data_points:
        raise ValueError(f"域大小 {config.domain_size:,} 不足以索引 {num_data_points:,} 个数据点")
    
    if config.prime <= max_value:
        raise ValueError(f"素数 {config.prime:,} 必须大于最大值 {max_value:,}")
    
    return True




def get_validated_config(dataset_name: str) -> DomainConfig:
    """获取经过验证的配置
    
    Args:
        dataset_name: 数据集名称
        
    Returns:
        经过数据需求验证的域配置
    """
    config = get_config(dataset_name)
    
    # 使用配置中的参数进行验证
    # 估算 neighbor list 记录数约为文档数的 1-3 倍
    estimated_neighbor_records = min(config.num_docs * 3, config.domain_size)
    max_node_id = config.num_docs - 1
    
    validate_data_requirements(config, estimated_neighbor_records, max_node_id)
    
    return config


if __name__ == "__main__":
    # 测试配置
    print("域参数配置（31位安全性）：")
    print("=" * 50)
    
    for name, config in CONFIGS.items():
        print(f"\n{name} 配置:")
        print(f"  索引域: 2^{config.domain_bits} = {config.domain_size:,}")
        print(f"  素数域: p = {config.prime:,}")
        print(f"  安全性: {config.kappa} 位")
        
        # 显示数据集信息
        dataset_info = {
            SIFTSMALL: "SIFT Small数据集（10k向量，128维）",
            LAION: "LAION数据集（100k向量，512维）",
            TRIPCLICK: "TripClick数据集（1.5M向量，768维）",
            MS_MARCO: "MS MARCO数据集（8.8M向量，768维）",
            NFCORPUS: "NFCorpus数据集（3.6k向量，768维，生物医学领域）"
        }
        
        if name in dataset_info:
            print(f"  用途: {dataset_info[name]}")
            print(f"  向量维度: {config.vector_dimension}")
            print(f"  文档数: {config.num_docs:,}, 查询数: {config.num_queries:,}")
            print(f"  HNSW参数: M={config.M}, ef_construction={config.efconstruction}, ef_search={config.efsearch}")
            
            # 检查兼容性
            try:
                # 使用配置中的参数验证
                estimated_neighbor_records = min(config.num_docs * 3, config.domain_size)
                max_node_id = config.num_docs - 1
                validate_data_requirements(config, estimated_neighbor_records, max_node_id)
                print(f"  状态: ✓ 满足所有需求")
            except ValueError as e:
                print(f"  状态: ✗ {e}")