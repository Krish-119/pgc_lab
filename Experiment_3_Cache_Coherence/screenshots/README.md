# Experiment 3 Screenshots Guide

Capture the following terminal screens for practical record submission:

| Filename | Content to Capture | Command to Run Before Snip |
|:---|:---|:---|
| `01_protocol_simulator.png` | Output of the 2-core MSI/MESI/MOESI simulator | `python3 src/coherence_sim.py 1000` |
| `02_compilation.png` | Successful compilation of C programs via Makefile | `make clean && make` |
| `03_false_sharing_execution.png` | 2-thread false sharing adjacent vs padded | `OMP_NUM_THREADS=2 ./false_sharing adjacent`<br>`OMP_NUM_THREADS=2 ./false_sharing padded` |
| `04_benchmark_tables.png` | Full output of Tables 1 and 2 | `./compare_numa` |
| `05_numa_hardware_topology.png` | System hardware NUMA nodes and CPU list | `numactl --hardware ; lscpu -e` |
