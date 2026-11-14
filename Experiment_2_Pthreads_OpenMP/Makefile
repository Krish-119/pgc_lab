CC = gcc
CFLAGS = -Wall -O2
PTHREAD_FLAGS = -pthread
OMP_FLAGS = -fopenmp

PTHREAD_TARGETS = thread1 thread2 thread_sum race mutex pthread_perf
OMP_TARGETS = omp1 omp_sum omp_race omp_critical omp_barrier omp_perf

all: $(PTHREAD_TARGETS) $(OMP_TARGETS)

thread1: src/thread1.c
	$(CC) $(CFLAGS) $(PTHREAD_FLAGS) -o $@ $<

thread2: src/thread2.c
	$(CC) $(CFLAGS) $(PTHREAD_FLAGS) -o $@ $<

thread_sum: src/thread_sum.c
	$(CC) $(CFLAGS) $(PTHREAD_FLAGS) -o $@ $<

race: src/race.c
	$(CC) $(CFLAGS) $(PTHREAD_FLAGS) -o $@ $<

mutex: src/mutex.c
	$(CC) $(CFLAGS) $(PTHREAD_FLAGS) -o $@ $<

pthread_perf: src/pthread_perf.c
	$(CC) $(CFLAGS) $(PTHREAD_FLAGS) -o $@ $<

omp1: src/omp1.c
	$(CC) $(CFLAGS) $(OMP_FLAGS) -o $@ $<

omp_sum: src/omp_sum.c
	$(CC) $(CFLAGS) $(OMP_FLAGS) -o $@ $<

omp_race: src/omp_race.c
	$(CC) $(CFLAGS) $(OMP_FLAGS) -o $@ $<

omp_critical: src/omp_critical.c
	$(CC) $(CFLAGS) $(OMP_FLAGS) -o $@ $<

omp_barrier: src/omp_barrier.c
	$(CC) $(CFLAGS) $(OMP_FLAGS) -o $@ $<

omp_perf: src/omp_perf.c
	$(CC) $(CFLAGS) $(OMP_FLAGS) -o $@ $<

clean:
	rm -f $(PTHREAD_TARGETS) $(OMP_TARGETS)

.PHONY: all clean
