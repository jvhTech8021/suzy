import os
import sys
import subprocess
import time
import shutil

def main():
    """Reset cache and restart the dashboard"""
    print("=== RESETTING CACHE AND RESTARTING DASHBOARD ===")
    
    # Kill any running dashboard processes
    print("Stopping any running dashboard processes...")
    subprocess.run("pkill -f 'python.*app.py'", shell=True)
    time.sleep(2)  # Wait for processes to terminate
    
    # Find and remove any cache files
    cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache-directory")
    if os.path.exists(cache_dir):
        print(f"Removing cache directory: {cache_dir}")
        shutil.rmtree(cache_dir)
    
    # Copy the champion profile files to a temp location and back
    # This sometimes helps with filesystem caching issues
    model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                            "march_madness_predictor/models/champion_profile/model")
    json_file = os.path.join(model_dir, "champion_profile.json")
    csv_file = os.path.join(model_dir, "all_teams_champion_profile.csv")
    
    print("Refreshing champion profile files...")
    if os.path.exists(json_file):
        with open(json_file, 'r') as f:
            json_content = f.read()
        with open(json_file, 'w') as f:
            f.write(json_content)
        os.system(f"touch {json_file}")
        print(f"Refreshed {json_file}")
    
    if os.path.exists(csv_file):
        with open(csv_file, 'r') as f:
            csv_content = f.read()
        with open(csv_file, 'w') as f:
            f.write(csv_content)
        os.system(f"touch {csv_file}")
        print(f"Refreshed {csv_file}")
    
    # Change ownership/permissions of the files
    print("Setting correct permissions...")
    os.system(f"chmod 644 {json_file}")
    os.system(f"chmod 644 {csv_file}")
    
    print("Starting dashboard...")
    subprocess.Popen(["python3", "march_madness_predictor/app.py"], 
                     cwd=os.path.dirname(os.path.abspath(__file__)))
    
    print("\nDashboard should now be running. Open your browser and visit:")
    print("http://127.0.0.1:8050/")
    print("\n=== PROCESS COMPLETE ===")

if __name__ == "__main__":
    main() 