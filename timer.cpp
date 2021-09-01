#include <time.h>

#if defined( _WIN32 )
#define NOMINMAX
#include <Windows.h>
#else
#include <sys/time.h>
#include <sys/types.h>
#include <unistd.h>
#endif

/*inline*/ unsigned long long getCurrentTick() {
  unsigned long long ret;
#if defined( _WIN32 )
  LARGE_INTEGER tick;
  ::QueryPerformanceCounter( &tick );               // Note: Windows times will need converting by the frequency
  ret = tick.QuadPart;
#else
  timespec now;
  // Look into CLOCK_MONOTONIC
  clock_gettime( CLOCK_REALTIME, &now );            // Note: Linux times are always in nanoseconds
  ret = (unsigned long long)now.tv_nsec + ( 1000000000ULL * (unsigned long long)now.tv_sec );
#endif
  return ret;
}

inline unsigned long long ticksPerSecond() {
  unsigned long long ret;
#if defined( _WIN32 )
  LARGE_INTEGER freq;
  ::QueryPerformanceFrequency( &freq );
  ret = freq.QuadPart;
#else
  ret = 1000000000ULL;
#endif
  return ret;
}

double delta_t( unsigned long long start, unsigned long long end ) {
  if( end == 0 ) end = getCurrentTick();
  double dticks = (double)end - (double)start;
  return dticks / (double)ticksPerSecond();
}

void cpuDelay(unsigned long long ticks) {
    unsigned long long now = getCurrentTick();
    while((getCurrentTick() - now) < ticks);
}

