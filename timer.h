#ifndef TIMER_H
#define TIMER_H

unsigned long long getCurrentTick();
unsigned long long ticksPerSecond();
double delta_t( unsigned long long start, unsigned long long end=0 );
void cpuDelay( unsigned long long ticks );
#endif

