#!/usr/bin/env python3
"""
[CN]
[CN] MPC、VDPF [CN]
"""

from dataclasses import dataclass
from typing import Dict, Any, List


@dataclass
class DomainConfig:
    """[CN]"""
    # [CN]
    domain_bits: int      # [CN]
    domain_size: int      # 2^domain_bits
    
    # [CN]
    kappa: int           # [CN] κ
    l: int               # [CN]（[CN] l = κ）
    
    # [CN]
    prime: int           # [CN] p
    
    # DPF [CN]
    output_bits: int     # DPF [CN]
    
    # [CN]
    vector_dimension: int # Vector dimension
    num_docs: int        # Number of documents[CN]
    num_queries: int     # Number of queries[CN]
    
    # HNSW [CN]
    efsearch: int        # [CN]
    efconstruction: int  # [CN]
    layer: int           # HNSW [CN]
    M: int               # [CN]connect[CN]
    
    def __post_init__(self):
        """[CN]"""
        assert self.domain_size == 2 ** self.domain_bits
        assert self.l == self.kappa, "[CN] l = κ"
        # [CN]：p ≈ 2^l
        gap_ratio = abs(self.prime - 2**self.l) / (2**self.l)
        assert gap_ratio < 0.5, f"[CN]：[CN] {gap_ratio} [CN]"
        

# [CN]
SIFTSMALL = "siftsmall"
LAION = "laion"
TRIPCLICK = "tripclick"
MS_MARCO = "ms_marco"
NFCORPUS = "nfcorpus"

# [CN]（[CN]31[CN]）
CONFIGS = {
 
    SIFTSMALL: DomainConfig(
        domain_bits=16,      # [CN] 65k neighbor list
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
        domain_bits=19,      # [CN] 524k neighbor list
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
        domain_bits=21,      # [CN] 2M neighbor list (1.5M[CN])
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
        domain_bits=24,      # [CN] 16M neighbor list (8.8M[CN])
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
        domain_bits=15,      # [CN] 32k neighbor list
        domain_size=32768,
        kappa=31,
        l=31,
        prime=2**31 - 1,     # 2,147,483,647
        output_bits=31,
        vector_dimension=768,  # pritamdeka/S-PubMedBert-MS-MARCO [CN]Vector dimension
        num_docs=3633,         # NFCorpus Number of documents[CN]
        num_queries=323,       # NFCorpus Number of queries[CN]
        efsearch=32,           # [CN]
        efconstruction=80,     # [CN]
        layer=2,               # [CN]Dataset2[CN]
        M=32                   # [CN]Dataset[CN] M
    )
}


def get_config(dataset_name: str = SIFTSMALL) -> DomainConfig:
    """[CN]
    
    Args:
        dataset_name: Dataset[CN]，[CN] SIFTSMALL
        
    Returns:
        [CN]Dataset[CN]
        
    Raises:
        ValueError: [CN]Dataset[CN]
    """
    if dataset_name not in CONFIGS:
        raise ValueError(f"[CN]Dataset[CN]: {dataset_name}. [CN]: {list_available_configs()}")
    return CONFIGS[dataset_name]


def list_available_configs() -> List[str]:
    """[CN]"""
    return list(CONFIGS.keys())


def validate_data_requirements(config: DomainConfig, num_data_points: int, max_value: int):
    """[CN]"""
    if config.domain_size < num_data_points:
        raise ValueError(f"[CN] {config.domain_size:,} [CN] {num_data_points:,} [CN]")
    
    if config.prime <= max_value:
        raise ValueError(f"[CN] {config.prime:,} [CN] {max_value:,}")
    
    return True




def get_validated_config(dataset_name: str) -> DomainConfig:
    """[CN]
    
    Args:
        dataset_name: Dataset[CN]
        
    Returns:
        [CN]
    """
    config = get_config(dataset_name)
    
    # [CN]
    # [CN] neighbor list [CN]Number of documents[CN] 1-3 [CN]
    estimated_neighbor_records = min(config.num_docs * 3, config.domain_size)
    max_node_id = config.num_docs - 1
    
    validate_data_requirements(config, estimated_neighbor_records, max_node_id)
    
    return config


if __name__ == "__main__":
    # [CN]
    print("[CN]（31[CN]）：")
    print("=" * 50)
    
    for name, config in CONFIGS.items():
        print(f"\n{name} [CN]:")
        print(f"  [CN]: 2^{config.domain_bits} = {config.domain_size:,}")
        print(f"  [CN]: p = {config.prime:,}")
        print(f"  [CN]: {config.kappa} [CN]")
        
        # [CN]Dataset[CN]
        dataset_info = {
            SIFTSMALL: "SIFT SmallDataset（10k[CN]，128[CN]）",
            LAION: "LAIONDataset（100k[CN]，512[CN]）",
            TRIPCLICK: "TripClickDataset（1.5M[CN]，768[CN]）",
            MS_MARCO: "MS MARCODataset（8.8M[CN]，768[CN]）",
            NFCORPUS: "NFCorpusDataset（3.6k[CN]，768[CN]，[CN]）"
        }
        
        if name in dataset_info:
            print(f"  [CN]: {dataset_info[name]}")
            print(f"  Vector dimension: {config.vector_dimension}")
            print(f"  Number of documents: {config.num_docs:,}, Number of queries: {config.num_queries:,}")
            print(f"  HNSW[CN]: M={config.M}, ef_construction={config.efconstruction}, ef_search={config.efsearch}")
            
            # [CN]
            try:
                # [CN]
                estimated_neighbor_records = min(config.num_docs * 3, config.domain_size)
                max_node_id = config.num_docs - 1
                validate_data_requirements(config, estimated_neighbor_records, max_node_id)
                print(f"  [CN]: ✓ [CN]")
            except ValueError as e:
                print(f"  [CN]: ✗ {e}")