#ifndef SVM_HPP
#define SVM_HPP

#include <inttypes.h>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <cstdlib>
#include <cmath>
#include <sys/time.h>
#include <assert.h>

#define VTYPE float

// Timing
static __inline__ uint64_t gettime(void) { 
  struct timeval tv; 
  gettimeofday(&tv, NULL); 
  return (((uint64_t)tv.tv_sec) * 1000000 + ((uint64_t)tv.tv_usec)); 
} 

static uint64_t usec;
__attribute__ ((noinline))  void begin_roi() { usec=gettime(); }
__attribute__ ((noinline)) void end_roi(const char* label) {
  usec = gettime() - usec;
  std::cout << label << " elapsed (sec): " << usec / 1000000.0 << "\n";
}

// Activation
__host__ __device__
inline VTYPE transfer(VTYPE x) {
    return (x >= 0) ? 1.0f : 0.0f;  // binary class output
}

#endif
