import sys
import pathlib
import hashlib
import tarfile
from tqdm import tqdm


def main():
    if len(sys.argv) != 7:
        print("Usage: python archive_datasets.py <list_file> <output_dir> <dataset_root> <task_id> <total_jobs> <type_name>")
        sys.exit(1)
    
    list_file = pathlib.Path(sys.argv[1])
    output_dir = pathlib.Path(sys.argv[2])
    dataset_root = pathlib.Path(sys.argv[3])
    task_id = int(sys.argv[4])  # 0-indexed
    total_jobs = int(sys.argv[5])
    type_name = sys.argv[6]  # e.g., "spoc_scenes"

    print(f"Reading directories from {list_file}...")
    assert list_file.is_file(), f"Error: {list_file} is not a file"
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Read all directories from the list file
    all_dirs = [line.strip() for line in list_file.read_text().splitlines() if line.strip()]
    print(f"Found {len(all_dirs)} total directories")
    
    # Filter directories for this task using hash-based distribution
    my_dirs = [
        x for x in all_dirs 
        if int(hashlib.md5((str(x) + "potato").encode()).hexdigest(), 16) % total_jobs == task_id
    ]
    
    num_dirs = len(my_dirs)
    print(f"Task {task_id}/{total_jobs} will process {num_dirs} directories")
    
    if num_dirs == 0:
        print("No directories to process for this shard.")
        return
    
    # Output tar file for this shard
    output_file = output_dir / f"{type_name}_shard_{task_id:03d}.tar"
    temp_output_file = output_dir / f"{type_name}_shard_{task_id:03d}.tar.tmp"
    
    # Skip if already exists
    if output_file.exists():
        print(f"Output file already exists: {output_file}")
        return
    
    print(f"Creating archive: {output_file}")
    print(f"Dataset root: {dataset_root}")
    
    # Create tar file with all directories
    success_count = 0
    error_count = 0
    
    with tarfile.open(temp_output_file, "w") as tar:
        for dir_path_str in tqdm(my_dirs):
            dir_path = pathlib.Path(dir_path_str)
            
            if not dir_path.exists():
                print(f"Skipping {dir_path}: directory does not exist")
                error_count += 1
                continue
            
            try:
                # Compute path relative to dataset root
                # e.g., /checkpoint/.../spoc/scenes/scene_name -> scenes/scene_name
                rel_path = dir_path.relative_to(dataset_root)
                
                # Use filter to exclude *.success files
                def exclude_success(tarinfo):
                    if tarinfo.name.endswith('.success'):
                        return None
                    return tarinfo
                
                tar.add(str(dir_path), arcname=str(rel_path), filter=exclude_success)
                success_count += 1
                
                if success_count % 100 == 0:
                    print(f"Added {success_count} directories...")
                    
            except Exception as e:
                print(f"Error adding {dir_path}: {e}")
                error_count += 1
    
    # Move temp file to final location
    temp_output_file.rename(output_file)
    
    print(f"Finished: {success_count} directories added, {error_count} errors")
    print(f"Output: {output_file}")


if __name__ == "__main__":
    main()
