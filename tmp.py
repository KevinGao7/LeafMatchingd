# cpu_stress.py
import multiprocessing as mp
import time
import math

def busy_loop():
    x = 0.0
    while True:
        x += math.sin(x) * math.cos(x)  # 纯计算，无IO

if __name__ == "__main__":
    n = mp.cpu_count()          # 当前系统核数
    print(f"[INFO] Detected {n} logical CPUs")
    print("[INFO] Launching stress workers...")
    procs = []
    for i in range(n):
        p = mp.Process(target=busy_loop)
        p.start()
        procs.append(p)

    time.sleep(10)              # 运行 10 秒
    print("[INFO] Done. Stopping workers.")
    for p in procs:
        p.terminate()
