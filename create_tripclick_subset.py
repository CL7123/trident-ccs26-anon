#!/usr/bin/env python3
import numpy as np
import struct
import os
from tqdm import tqdm
import time

def read_fvecs(filename, indices=None):
    """Read fvecs file, optionally read vectors at specified indices"""
    vectors = []
    with open(filename, 'rb') as f:
        if indices is None:
            # Read all vectors
            while True:
                dim_bytes = f.read(4)
                if not dim_bytes:
                    break
                dim = struct.unpack('i', dim_bytes)[0]
                vector = np.frombuffer(f.read(dim * 4), dtype=np.float32)
                vectors.append(vector)
        else:
            # Read vectors at specified indices
            indices = sorted(indices)
            current_idx = 0
            dim = struct.unpack('i', f.read(4))[0]
            f.seek(0)

            for target_idx in indices:
                # Skip to target position
                while current_idx < target_idx:
                    f.seek(4, 1)  # Skip dimension
                    f.seek(dim * 4, 1)  # Skip vector data
                    current_idx += 1

                # Read target vector
                dim = struct.unpack('i', f.read(4))[0]
                vector = np.frombuffer(f.read(dim * 4), dtype=np.float32)
                vectors.append(vector)
                current_idx += 1
                
    return np.array(vectors)

def write_fvecs(filename, vectors):
    """Write to fvecs file"""
    with open(filename, 'wb') as f:
        for vector in vectors:
            dim = len(vector)
            f.write(struct.pack('i', dim))
            f.write(vector.astype(np.float32).tobytes())

def read_ivecs(filename):
    """Read ivecs file"""
    with open(filename, 'rb') as f:
        vectors = []
        while True:
            dim_bytes = f.read(4)
            if not dim_bytes:
                break
            dim = struct.unpack('i', dim_bytes)[0]
            vector = np.frombuffer(f.read(dim * 4), dtype=np.int32)
            vectors.append(vector)
    return np.array(vectors)

def write_ivecs(filename, vectors):
    """Write to ivecs file"""
    with open(filename, 'wb') as f:
        for vector in vectors:
            dim = len(vector)
            f.write(struct.pack('i', dim))
            f.write(vector.astype(np.int32).tobytes())

def create_subset(base_path, output_path, subset_ratio=0.1):
    """Create a data subset, preserving all ground truth vectors"""
    print(f"Creating {subset_ratio*100:.0f}% subset of TripClick dataset...")
    print(f"Strategy: Keep all ground truth vectors + random sampling\n")

    # 1. Read ground truth to get required vector indices
    print("1. Reading ground truth...")
    gt = read_ivecs(os.path.join(base_path, 'gt.ivecs'))
    gt_indices = np.unique(gt.flatten())
    print(f"   Found {len(gt_indices):,} unique vectors in ground truth")

    # 2. Calculate required number of vectors
    with open(os.path.join(base_path, 'base.fvecs'), 'rb') as f:
        dim = struct.unpack('i', f.read(4))[0]
        f.seek(0, 2)
        file_size = f.tell()
        vector_size = 4 + dim * 4
        total_vectors = file_size // vector_size

    target_count = int(total_vectors * subset_ratio)
    additional_needed = target_count - len(gt_indices)

    print(f"\n2. Calculating subset size:")
    print(f"   Total base vectors: {total_vectors:,}")
    print(f"   Target subset size: {target_count:,} ({subset_ratio*100:.0f}%)")
    print(f"   Additional vectors needed: {additional_needed:,}")

    # 3. Select additional vectors
    print(f"\n3. Selecting additional vectors...")
    all_indices = set(range(total_vectors))
    gt_indices_set = set(gt_indices.tolist())
    available_indices = list(all_indices - gt_indices_set)

    # Randomly select additional vectors
    np.random.seed(42)  # Fixed random seed to ensure reproducibility
    additional_indices = np.random.choice(available_indices, additional_needed, replace=False)

    # Merge all selected indices
    selected_indices = np.concatenate([gt_indices, additional_indices])
    selected_indices = np.sort(selected_indices)

    print(f"   Total selected vectors: {len(selected_indices):,}")

    # Create output directory
    os.makedirs(output_path, exist_ok=True)

    # 4. Create index mapping
    print("\n4. Creating index mapping...")
    old_to_new = {old_idx: new_idx for new_idx, old_idx in enumerate(selected_indices)}

    # 5. Read and write selected base vectors
    print("\n5. Writing base vectors...")
    start_time = time.time()

    with open(os.path.join(base_path, 'base.fvecs'), 'rb') as f_in:
        with open(os.path.join(output_path, 'base.fvecs'), 'wb') as f_out:
            current_idx = 0

            for target_idx in tqdm(selected_indices, desc="Writing vectors"):
                # Skip to target position
                while current_idx < target_idx:
                    dim_bytes = f_in.read(4)
                    if not dim_bytes:
                        break
                    dim = struct.unpack('i', dim_bytes)[0]
                    f_in.seek(dim * 4, 1)
                    current_idx += 1

                # Read and write target vector
                dim_bytes = f_in.read(4)
                vector_bytes = f_in.read(dim * 4)
                f_out.write(dim_bytes)
                f_out.write(vector_bytes)
                current_idx += 1

    print(f"   Time taken: {time.time() - start_time:.2f} seconds")

    # 6. Copy query vectors
    print("\n6. Copying query vectors...")
    query_vectors = read_fvecs(os.path.join(base_path, 'query.fvecs'))
    write_fvecs(os.path.join(output_path, 'query.fvecs'), query_vectors)
    print(f"   Copied {len(query_vectors)} query vectors")

    # 7. Update ground truth indices
    print("\n7. Updating ground truth indices...")
    new_gt = []
    for query_gt in gt:
        new_indices = [old_to_new[old_idx] for old_idx in query_gt]
        new_gt.append(new_indices)

    new_gt = np.array(new_gt)
    write_ivecs(os.path.join(output_path, 'gt.ivecs'), new_gt)

    # 8. Verify results
    print("\n8. Verification:")
    print(f"   All ground truth indices valid: {np.all(new_gt >= 0) and np.all(new_gt < len(selected_indices))}")
    print(f"   Ground truth vectors preserved: {len(gt_indices)} / {len(gt_indices)}")

    print(f"\nSubset created successfully in: {output_path}")
    print(f"  - Base vectors: {len(selected_indices):,}")
    print(f"  - Query vectors: {len(query_vectors)}")
    print(f"  - Ground truth: {len(new_gt)} queries × {new_gt.shape[1]} neighbors")

if __name__ == "__main__":
    base_path = "~/trident/dataset/tripclick"
    output_path = "~/trident/dataset/tripclick_subset_10"
    
    create_subset(base_path, output_path, subset_ratio=0.1)