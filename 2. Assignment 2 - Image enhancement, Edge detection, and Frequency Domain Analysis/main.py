from utils import ensure_dirs
import sys
import time
import os

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def main():
    ensure_dirs()

    print("DIP Homework 2")
    # print(f"  Python executable : {sys.executable}")
    # print(f"  Working directory : {ROOT}")

    results = {}
    tasks = [1, 2, 3]

    for task_num in tasks:
        print(f"TASK {task_num}")
        t_start = time.time()

        if task_num == 1:
            from section1 import run_section1

            run_section1()
        elif task_num == 2:
            from section2 import run_section2

            run_section2()
        elif task_num == 3:
            from section3 import run_section3

            run_section3()

        elapsed = time.time() - t_start
        results[task_num] = elapsed
        print(f"\n  Done  Task {task_num} finished in {elapsed:.2f} seconds")

    print("SUMMARY")
    total = sum(results.values())
    for task_num, elapsed in results.items():
        print(f"  Task {task_num} : {elapsed:8.2f} s")
    print(f"  {'─'*20}")
    print(f"  Total   : {total:8.2f} s")
    print(f"\n  Output folders:")
    print(f"    output/section1/   Edge Detection results")
    print(f"    output/section2/   Noise & Restoration results")
    print(f"    output/section3/   Enhancement & Frequency Domain results")
    print()


if __name__ == "__main__":
    main()
