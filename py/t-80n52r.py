#!/usr/bin/env python3

import subprocess
import sys
import os
import time
import argparse

def run_script(script_path):
    """
    Runs a Python script and returns its success/failure status
    
    Args:
        script_path (str): Path to the Python script
        
    Returns:
        bool: True if script ran successfully, False otherwise
    """
    print(f"\n{'='*50}")
    print(f"Running: {script_path}")
    print(f"{'='*50}")
    
    try:
        # Get the same Python interpreter that's running this script
        python_executable = sys.executable
        
        # Run the script with the current Python interpreter
        result = subprocess.run(
            [python_executable, script_path],
            check=True,
            text=True
        )
        
        print(f"\n✅ {script_path} completed successfully")
        return True
    
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Error running {script_path}")
        print(f"Exit code: {e.returncode}")
        return False
    
    except FileNotFoundError:
        print(f"\n❌ Script not found: {script_path}")
        return False

def parse_arguments():
    """
    Parse command line arguments
    
    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description='Run multiple Python scripts in sequence.'
    )
    
    parser.add_argument(
        '--order', 
        nargs='+',
        type=int,
        help='Specify which scripts to run and their order. Example: --order 0 2 3'
    )
    
    parser.add_argument(
        '--prefix',
        type=str,
        default='part',
        help='Prefix for script names. Default: "part"'
    )
    
    parser.add_argument(
        '--suffix',
        type=str,
        default='.py',
        help='Suffix for script names. Default: ".py"'
    )
    
    return parser.parse_args()

def main():
    # Parse command-line arguments
    args = parse_arguments()
    
    # Default script order if not specified
    default_order = [0, 1, 2, 3]
    
    # Use the specified order or default
    script_indices = args.order if args.order else default_order
    
    # Create the list of scripts to run based on the order
    scripts = [f"{args.prefix}{index}{args.suffix}" for index in script_indices]
    
    print(f"Starting script execution sequence at {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Scripts to run: {', '.join(scripts)}")
    
    # Track success/failure for summary
    results = {}
    
    # Run each script in succession
    for script in scripts:
        success = run_script(script)
        results[script] = success
        
        # Optional: add a small delay between scripts
        if success:
            time.sleep(1)
    
    # Print summary
    print("\n" + "="*50)
    print("EXECUTION SUMMARY")
    print("="*50)
    
    all_success = True
    for script, success in results.items():
        status = "✅ SUCCESS" if success else "❌ FAILED"
        print(f"{script}: {status}")
        if not success:
            all_success = False
    
    # Final status
    print("\nFinal status:", "✅ All scripts completed successfully" if all_success else "❌ Some scripts failed")
    
    # Return appropriate exit code
    return 0 if all_success else 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)