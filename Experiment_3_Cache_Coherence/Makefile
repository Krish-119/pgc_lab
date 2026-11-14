CC = gcc
CFLAGS = -O2 -Wall
OMP_FLAGS = -fopenmp

TARGETS = false_sharing numa_sweep compare_numa

all: $(TARGETS)

false_sharing: src/false_sharing.c
	$(CC) $(CFLAGS) $(OMP_FLAGS) -o $@ $<

numa_sweep: src/numa_sweep.c
	$(CC) $(CFLAGS) $(OMP_FLAGS) -o $@ $<

compare_numa: src/compare_numa.c
	$(CC) $(CFLAGS) $(OMP_FLAGS) -o $@ $<

clean:
	rm -f $(TARGETS)

.PHONY: all clean
