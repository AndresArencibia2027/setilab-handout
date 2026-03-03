
#define _GNU_SOURCE
#include <stdlib.h>
#include <stdio.h>
#include <ctype.h>
#include <assert.h>

#include "filter.h"
#include "signal.h"
#include "timing.h"

#include <sched.h>    // for processor affinity
#include <unistd.h>   // unix standard apis
#include <pthread.h>  // pthread api

int vector_len;       // length of vector we will sum
double* vector;       // the vector we will sum

int num_threads;            // number of threads we will use
int num_processors;         // number of processors we will use
pthread_t* tid;             // array of thread ids
double* partial_sum;        // partial sums, one for each processor

#define MAXWIDTH 40
#define THRESHOLD 2.0
#define ALIENS_LOW  50000.0
#define ALIENS_HIGH 150000.0

struct args {
    long tid; 
    signal* sig; 
    int filter_order;
    int bandwidth;
    double* filter_coeffs;
    double* band_power;
    int vector_len;
};


void usage() {
  printf("usage: band_scan text|bin|mmap signal_file Fs filter_order num_bands\n");
}

double avg_power(double* data, int num) {

  double ss = 0;
  for (int i = 0; i < num; i++) {
    ss += data[i] * data[i];
  }

  return ss / num;
}

double max_of(double* data, int num) {

  double m = data[0];
  for (int i = 1; i < num; i++) {
    if (data[i] > m) {
      m = data[i];
    }
  }
  return m;
}

double avg_of(double* data, int num) {

  double s = 0;
  for (int i = 0; i < num; i++) {
    s += data[i];
  }
  return s / num;
}

void remove_dc(double* data, int num) {

  double dc = avg_of(data,num);

  printf("Removing DC component of %lf\n",dc);

  for (int i = 0; i < num; i++) {
    data[i] -= dc;
  }
}

// Function run by each thread
void* worker(void* varg) {
  struct args* arg = (struct args*)varg;
  long myid     = (long)arg->tid;

  int blocksize = arg->vector_len / num_threads; // note: floor

  // put ourselves on the desired processor
  cpu_set_t set;
  CPU_ZERO(&set);
  CPU_SET(myid % num_processors, &set);
  if (sched_setaffinity(0, sizeof(set), &set) < 0) { // do it
    perror("Can't setaffinity"); // hopefully doesn't fail
    exit(-1);
  }

  // This figures out the chunk of the vector I should
  // work on based on my id
  int mystart = myid * blocksize;
  int myend   = 0;
  if (myid == (num_threads - 1)) { // last thread
    // the last thread will take care of the leftover
    // elements of the vector, in case num_threads doesn't
    // divide vector_len
    // WARNING: this is a suboptimal solution. It means that the last thread
    // might do much more work than the other threads (up to almost double)
    // which will slow down the entire job. A better solution would split up
    // remainder work equally between threads...
    myend = arg->vector_len;
  } else {
    myend = (myid + 1) * blocksize;
  }

    for (int band = mystart; band < myend; band++) {
        // Make the filter
        generate_band_pass(arg->sig->Fs,
                        band * arg->bandwidth + 0.0001, // keep within limits
                        (band + 1) * arg->bandwidth - 0.0001,
                        arg->filter_order,
                        arg->filter_coeffs);
        hamming_window(arg->filter_order,arg->filter_coeffs);

        // Convolve
        convolve_and_compute_power(arg->sig->num_samples,
                                arg->sig->data,
                                arg->filter_order,
                                arg->filter_coeffs,
                                &(arg->band_power[band]));

  }

  pthread_exit(NULL);   

}


int analyze_signal(signal* sig, int filter_order, int num_bands, double* lb, double* ub, int num_threads, int num_processors) {

  double Fc        = (sig->Fs) / 2;
  double bandwidth = Fc / num_bands;

  remove_dc(sig->data,sig->num_samples);

  double signal_power = avg_power(sig->data,sig->num_samples);

  printf("signal average power:     %lf\n", signal_power);

  resources rstart;
  get_resources(&rstart,THIS_PROCESS);
  double start = get_seconds();
  unsigned long long tstart = get_cycle_count();

  double filter_coeffs[filter_order + 1];
  double band_power[num_bands];

  /////////////////////////////// PARALLEL SUM CODE

  // launch threads
  // thread i will compute partial_sum[i], which will sum
  // vector[i*ceiling(vector_size/num_threads)
  //          through
  //        (i+1)*floor(vector_size/num_threads) ]
  // the last thread will also handle the additional elements

  struct args* arg;
  arg = (struct args*)malloc(sizeof(struct args) * num_threads); 

  for (long i = 0; i < num_threads; i++) {
      arg[i].band_power = band_power;
      arg[i].bandwidth = bandwidth;
      arg[i].filter_coeffs = filter_coeffs;
      arg[i].filter_order = filter_order;
      arg[i].sig = sig;
      arg[i].tid = i;
      arg[i].vector_len = num_bands;

      double local_coeffs[arg->filter_order + 1];

    int returncode = pthread_create(&(tid[i]),  // thread id gets put here
                                    NULL, // use default attributes
                                    worker, // thread will begin in this function
                                    &arg[i] // we'll give it i as the argument
                                    );
    if (returncode != 0) {
      perror("Failed to start thread");
      exit(-1);
    }
  }

  // now we will join all the threads
  for (int i = 0; i < num_threads; i++) {
    int returncode = pthread_join(tid[i], NULL);
    if (returncode != 0) {
      perror("join failed");
      exit(-1);
    }
  }

  for (int i = 0; i < num_bands; i++){
    printf("%f\n",band_power[i]);
  }

  unsigned long long tend = get_cycle_count();
  double end = get_seconds();

  resources rend;
  get_resources(&rend,THIS_PROCESS);

  resources rdiff;
  get_resources_diff(&rstart, &rend, &rdiff);

  // Pretty print results
  double max_band_power = max_of(band_power,num_bands);
  double avg_band_power = avg_of(band_power,num_bands);
  int wow = 0;
  *lb = -1;
  *ub = -1;

  for (int band = 0; band < num_bands; band++) {
    double band_low  = band * bandwidth + 0.0001;
    double band_high = (band + 1) * bandwidth - 0.0001;

    printf("%5d %20lf to %20lf Hz: %20lf ",
           band, band_low, band_high, band_power[band]);

    for (int i = 0; i < MAXWIDTH * (band_power[band] / max_band_power); i++) {
      printf("*");
    }

    if ((band_low >= ALIENS_LOW && band_low <= ALIENS_HIGH) ||
        (band_high >= ALIENS_LOW && band_high <= ALIENS_HIGH)) {

      // band of interest
      if (band_power[band] > THRESHOLD * avg_band_power) {
        printf("(WOW)");
        wow = 1;
        if (*lb < 0) {
          *lb = band * bandwidth + 0.0001;
        }
        *ub = (band + 1) * bandwidth - 0.0001;
      } else {
        printf("(meh)");
      }
    } else {
      printf("(meh)");
    }

    printf("\n");
  }

  printf("Resource usages:\n\
User time        %lf seconds\n\
System time      %lf seconds\n\
Page faults      %ld\n\
Page swaps       %ld\n\
Blocks of I/O    %ld\n\
Signals caught   %ld\n\
Context switches %ld\n",
         rdiff.usertime,
         rdiff.systime,
         rdiff.pagefaults,
         rdiff.pageswaps,
         rdiff.ioblocks,
         rdiff.sigs,
         rdiff.contextswitches);

  printf("Analysis took %llu cycles (%lf seconds) by cycle count, timing overhead=%llu cycles\n"
         "Note that cycle count only makes sense if the thread stayed on one core\n",
         tend - tstart, cycles_to_seconds(tend - tstart), timing_overhead());
  printf("Analysis took %lf seconds by basic timing\n", end - start);

  return wow;
}


//main
int main(int argc, char* argv[]) {

  if (argc != 8) {
    usage();
    return -1;
  }

    char sig_type    = toupper(argv[1][0]);
    char* sig_file   = argv[2];
    double Fs        = atof(argv[3]);
    int filter_order = atoi(argv[4]);
    int num_bands    = atoi(argv[5]);
  
    num_threads = atoi(argv[6]); // number of threads
    num_processors = atoi(argv[7]); // numer of processors to use

    tid         = (pthread_t*)malloc(sizeof(pthread_t) * num_threads);

    assert(Fs > 0.0);
    assert(filter_order > 0 && !(filter_order & 0x1));
    assert(num_bands > 0);

    printf("type:     %s\n\
        file:     %s\n\
        Fs:       %lf Hz\n\
        order:    %d\n\
        bands:    %d\n",
                sig_type == 'T' ? "Text" : (sig_type == 'B' ? "Binary" : (sig_type == 'M' ? "Mapped Binary" : "UNKNOWN TYPE")),
                sig_file,
                Fs,
                filter_order,
                num_bands);

  printf("Load or map file\n");

  signal* sig;
  switch (sig_type) {
    case 'T':
      sig = load_text_format_signal(sig_file);
      break;

    case 'B':
      sig = load_binary_format_signal(sig_file);
      break;

    case 'M':
      sig = map_binary_format_signal(sig_file);
      break;

    default:
      printf("Unknown signal type\n");
      return -1;
  }

  if (!sig) {
    printf("Unable to load or map file\n");
    return -1;
  }

  sig->Fs = Fs;

  double start = 0;
  double end   = 0;


//original analyze_signal
  if (analyze_signal(sig, filter_order, num_bands, &start, &end, num_threads, num_processors)) {
    printf("POSSIBLE ALIENS %lf-%lf HZ (CENTER %lf HZ)\n", start, end, (end + start) / 2.0);
  } else {
    printf("no aliens\n");
  }

  free_signal(sig);

  return 0;
}
