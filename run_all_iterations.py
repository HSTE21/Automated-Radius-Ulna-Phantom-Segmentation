import os
import subprocess
import sys
import time

def run_script(script_path):
    """Runs a python script and waits for it to finish."""
    print(f"\n{'='*60}")
    print(f"RUNNING: {script_path}")
    print(f"{'='*60}")
    
    # Get the directory of the script to run it from its local context
    script_dir = os.path.dirname(os.path.abspath(script_path))
    script_name = os.path.basename(script_path)
    
    start_time = time.time()
    try:
        # Use sys.executable to ensure we use the same python environment
        result = subprocess.run(
            [sys.executable, script_name],
            cwd=script_dir,
            capture_output=False, # Show output in real-time
            text=True
        )
        duration = time.time() - start_time
        if result.returncode == 0:
            print(f"\nSUCCESS: {script_name} finished in {duration:.2f}s")
            return True
        else:
            print(f"\nERROR: {script_name} failed with return code {result.returncode}")
            return False
    except Exception as e:
        print(f"\nEXCEPTION: Could not run {script_name}: {e}")
        return False

def main():
    root_dir = os.getcwd()
    
    # Define the tasks (bone iterations and skin segmentation) in logical order
    tasks = [
        "Iteration 01 - Baseline Discrete 4D Grid Search",
        "Iteration 02 - Automated Bayesian Optimization (TPE)",
        "Iteration 03 - ROI-Restricted Precision Tuning",
        "Iteration 04 - Multi-Otsu Statistical Thresholding",
        "Iteration 05 - Sauvola Local Adaptive Thresholding",
        "Iteration 06 - 3D Morphological Active Contours",
        "Iteration 07 - Marker-Controlled Watershed",
        "Iteration 08 - 2D Slice-Wise Regional Active Contours",
        "Iteration 09 - Diffusion-Based Random Walker",
        "Iteration 10 - Canny Edge-Based Volume Reconstruction",
        "Skin Segmentation" # Dedicated task, not a bone iteration
    ]
    
    all_scripts = []
    
    # Build the list of scripts to run
    for task_folder in tasks:
        it_path = os.path.join(root_dir, task_folder)
        if not os.path.exists(it_path):
            continue
            
        # Check for sub-runs (Iteration 2 & 3)
        subdirs = [d for d in os.listdir(it_path) if os.path.isdir(os.path.join(it_path, d)) and not d.startswith('.')]
        
        if subdirs:
            # Sort subdirs (e.g., Run 1, Run 2...)
            subdirs.sort()
            for subdir in subdirs:
                run_path = os.path.join(it_path, subdir)
                scripts = [f for f in os.listdir(run_path) if f.endswith('.py')]
                # Search scripts first, then final segmentation
                search = [f for f in scripts if 'search' in f or 'grid' in f]
                final = [f for f in scripts if 'final_segmentation' in f or 'skin_segmentation' in f]
                for s in search: all_scripts.append(os.path.join(run_path, s))
                for f in final: all_scripts.append(os.path.join(run_path, f))
        else:
            scripts = [f for f in os.listdir(it_path) if f.endswith('.py')]
            search = [f for f in scripts if 'search' in f or 'grid' in f]
            final = [f for f in scripts if 'final_segmentation' in f or 'skin_segmentation' in f]
            for s in search: all_scripts.append(os.path.join(it_path, s))
            for f in final: all_scripts.append(os.path.join(it_path, f))

    print(f"Found {len(all_scripts)} scripts to execute.")
    
    failed_scripts = []
    total_start = time.time()
    
    for script in all_scripts:
        success = run_script(script)
        if not success:
            failed_scripts.append(script)
            
    total_duration = time.time() - total_start
    
    print(f"\n{'-'*60}")
    print(f"ALL TASKS COMPLETED IN {total_duration/60:.2f} MINUTES")
    print(f"{'-'*60}")
    
    if failed_scripts:
        print("\nThe following scripts failed:")
        for f in failed_scripts:
            print(f"  - {f}")
    else:
        print("\nAll scripts executed successfully!")

if __name__ == "__main__":
    main()
