import pickle
import os
import time
import hashlib

def get_file_hash(file_path):
    """Calculate MD5 hash of a file"""
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

def load_solved_states(cache_dir='cache', cache_file='solved_states_cache.pkl'):
    """
    Load solved_states from cache if available and valid, otherwise load from original file
    """
    # Ensure cache directory exists
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, cache_file)
    
    # Check if valid cache exists
    if os.path.exists(cache_path):
        # Verify cache integrity
        try:
            with open(cache_path, 'rb') as f:
                # Load metadata first
                metadata = pickle.load(f)
                # Verify source file hash
                if metadata['source_hash'] == get_file_hash('solutions_dict.pkl'):
                    print("Loading solved_states from cache...")
                    start_time = time.time()
                    solved_states = pickle.load(f)
                    print(f"Loaded in {time.time() - start_time:.2f} seconds")
                    return solved_states
        except (pickle.PickleError, EOFError, KeyError) as e:
            print(f"Cache validation failed: {e}. Rebuilding cache...")
    
    # If no valid cache, load from original file
    print("Loading solved_states from original file...")
    start_time = time.time()
    with open('solutions_dict.pkl', 'rb') as f:
        solved_states = pickle.load(f)
    print(f"Loaded in {time.time() - start_time:.2f} seconds")
    
    # Save to cache with metadata
    print("Caching solved_states...")
    with open(cache_path, 'wb') as f:
        # Store metadata first
        pickle.dump({
            'source_hash': get_file_hash('solutions_dict.pkl'),
            'timestamp': time.time()
        }, f, protocol=pickle.HIGHEST_PROTOCOL)
        # Then store the actual data
        pickle.dump(solved_states, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    return solved_states