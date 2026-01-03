#!/usr/bin/env python3
"""
Unified domain parameter configuration
Ensures consistent parameters across MPC, VDPF, and tests
"""

from dataclasses import dataclass
from typing import Dict, Any, List


@dataclass
class DomainConfig:
    """Domain configuration parameters"""
    # Index domain parameters
    domain_bits: int      # Number of bits for domain size
    domain_size: int      # 2^domain_bits

    # Security parameters
    kappa: int           # Security parameter κ
    l: int               # Binary domain parameter (usually l = κ)

    # Prime field parameters
    prime: int           # Prime p

    # DPF output parameters
    output_bits: int     # Number of bits for DPF output

    # Vector and query parameters
    vector_dimension: int # Vector dimension
    num_docs: int        # Document count
    num_queries: int     # Query count

    # HNSW index parameters
    efsearch: int        # Candidate count during search
    efconstruction: int  # Candidate count during construction
    layer: int           # HNSW layer count
    M: int               # Maximum connections per layer

    def __post_init__(self):
        """Validate parameter consistency"""
        assert self.domain_size == 2 ** self.domain_bits
        assert self.l == self.kappa, "Current design requires l = κ"
        # Validate security requirement: p ≈ 2^l
        gap_ratio = abs(self.prime - 2**self.l) / (2**self.l)
        assert gap_ratio < 0.5, f"Prime selection insecure: gap ratio {gap_ratio} too large"


# Configuration name constants
SIFTSMALL = "siftsmall"
LAION = "laion"
TRIPCLICK = "tripclick"
MS_MARCO = "ms_marco"
NFCORPUS = "nfcorpus"

# Predefined configurations (31-bit security only)
CONFIGS = {
 
    SIFTSMALL: DomainConfig(
        domain_bits=16,      # support 65k neighbor list
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
        domain_bits=19,      # support 524k neighbor list
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
        domain_bits=21,      # support 2M neighbor list (1.5M requirement)
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
        domain_bits=24,      # support 16M neighbor list (8.8M requirement)
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
        domain_bits=15,      # support 32k neighbor list
        domain_size=32768,
        kappa=31,
        l=31,
        prime=2**31 - 1,     # 2,147,483,647
        output_bits=31,
        vector_dimension=768,  # Vector dimension for pritamdeka/S-PubMedBert-MS-MARCO
        num_docs=3633,         # NFCorpus document count
        num_queries=323,       # NFCorpus query count
        efsearch=32,           # Can keep default
        efconstruction=80,     # Can keep default
        layer=2,               # 2 layers sufficient for small dataset
        M=32                   # Small dataset can use smaller M
    )
}


def get_config(dataset_name: str = SIFTSMALL) -> DomainConfig:
    """Unified configuration retrieval interface

    Args:
        dataset_name: Dataset name, defaults to SIFTSMALL

    Returns:
        Domain configuration for the corresponding dataset

    Raises:
        ValueError: When dataset name does not exist
    """
    if dataset_name not in CONFIGS:
        raise ValueError(f"Unknown dataset configuration: {dataset_name}. Available configurations: {list_available_configs()}")
    return CONFIGS[dataset_name]


def list_available_configs() -> List[str]:
    """List all available configuration names"""
    return list(CONFIGS.keys())


def validate_data_requirements(config: DomainConfig, num_data_points: int, max_value: int):
    """Validate whether configuration meets data requirements"""
    if config.domain_size < num_data_points:
        raise ValueError(f"Domain size {config.domain_size:,} insufficient to index {num_data_points:,} data points")

    if config.prime <= max_value:
        raise ValueError(f"Prime {config.prime:,} must be greater than maximum value {max_value:,}")

    return True




def get_validated_config(dataset_name: str) -> DomainConfig:
    """Get validated configuration

    Args:
        dataset_name: Dataset name
        
    Returns:
        Domain configuration validated against data requirements
    """
    config = get_config(dataset_name)

    # Validate using parameters from configuration
    # Estimate neighbor list records to be about 1-3 times document count
    estimated_neighbor_records = min(config.num_docs * 3, config.domain_size)
    max_node_id = config.num_docs - 1

    validate_data_requirements(config, estimated_neighbor_records, max_node_id)

    return config


if __name__ == "__main__":
    # Test configuration
    print("Domain Parameter Configuration (31-bit security):")
    print("=" * 50)

    for name, config in CONFIGS.items():
        print(f"\n{name} configuration:")
        print(f"  Index domain: 2^{config.domain_bits} = {config.domain_size:,}")
        print(f"  Prime field: p = {config.prime:,}")
        print(f"  Security: {config.kappa} bit")

        # Show dataset information
        dataset_info = {
            SIFTSMALL: "SIFT Small dataset (10k vectors, 128 dim)",
            LAION: "LAION dataset (100k vectors, 512 dim)",
            TRIPCLICK: "TripClick dataset (1.5M vectors, 768 dim)",
            MS_MARCO: "MS MARCO dataset (8.8M vectors, 768 dim)",
            NFCORPUS: "NFCorpus dataset (3.6k vectors, 768 dim, biomedical domain)"
        }

        if name in dataset_info:
            print(f"  Purpose: {dataset_info[name]}")
            print(f"  Vector dimension: {config.vector_dimension}")
            print(f"  Documents: {config.num_docs:,}, Queries: {config.num_queries:,}")
            print(f"  HNSW parameters: M={config.M}, ef_construction={config.efconstruction}, ef_search={config.efsearch}")

            # Check compatibility
            try:
                # Validate using parameters from configuration
                estimated_neighbor_records = min(config.num_docs * 3, config.domain_size)
                max_node_id = config.num_docs - 1
                validate_data_requirements(config, estimated_neighbor_records, max_node_id)
                print(f"  Status: Meets all requirements")
            except ValueError as e:
                print(f"  Status: {e}")