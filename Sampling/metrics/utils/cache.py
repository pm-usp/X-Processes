import pm4py
import pickle
import os


def load_log_with_cache(log_path: str, cache_dir: str):
    """
    Loads a PM4Py EventLog. If a cached version (.pkl) exists in the specified
    cache directory, it's used. Otherwise, it reads the .xes file, creates a
    cache, and returns the log.

    Args:
        log_path (str): The full path to the .xes log file.
        cache_dir (str): The path to the directory where cache files are stored.
    """

    base_name = os.path.basename(log_path)
    file_id = os.path.splitext(base_name)[0]
    cache_path = os.path.join(cache_dir, f"{file_id}.pkl")

    # Check if the cached file exists
    if os.path.exists(cache_path):
        print(f"Loading '{file_id}' from fast cache...")
        with open(cache_path, 'rb') as f:
            log = pickle.load(f)
        return log

    # If no cache, read the .xes file
    else:
        print(f"No cache found for '{file_id}'. Reading .xes file...")
        log = pm4py.read_xes(log_path)

        print(f"Saving '{file_id}' to cache for future runs...")
        os.makedirs(cache_dir, exist_ok=True)
        with open(cache_path, 'wb') as f:
            pickle.dump(log, f)

        return log
    #endif
#enddef