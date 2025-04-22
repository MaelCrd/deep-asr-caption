import shutil
import time
import os
from datetime import datetime

def copy_file(src, dst):
    if os.path.isfile(src):
        shutil.copy2(src, dst)
        print(f"Copied {src} to {dst}")
    else:
        print(f"Source file {src} does not exist.")

def main():
    src = 'asr_v2.pth'
    
    while True:
        now = datetime.now()
        dst = f'asr_v2_copy-{now.strftime("%Y-%m-%d-%H-%M-%S")}.pth'
        copy_file(src, dst)
        time.sleep(3600*2)  # Sleep for 1 hour

if __name__ == "__main__":
    main()