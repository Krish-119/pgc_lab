/*
 * Experiment 2: Multithreaded Programming Using Pthreads and OpenMP
 * File: thread1.c
 * Description: Pthread creation and termination using pthread_create() and pthread_join().
 */

#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>

void *thread_function(void *arg) {
    (void)arg;
    printf("Hello from the thread!\n");
    pthread_exit(NULL);
}

int main(void) {
    pthread_t thread_id;
    int status;

    printf("Main thread: creating an additional thread...\n");
    status = pthread_create(&thread_id, NULL, thread_function, NULL);
    if (status != 0) {
        fprintf(stderr, "Error creating thread: %d\n", status);
        return EXIT_FAILURE;
    }

    status = pthread_join(thread_id, NULL);
    if (status != 0) {
        fprintf(stderr, "Error joining thread: %d\n", status);
        return EXIT_FAILURE;
    }

    printf("Main thread finished.\n");
    return EXIT_SUCCESS;
}
