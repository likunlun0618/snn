#include <sys/time.h>
#include "time.h"

long time()
{
    struct timeval t;
    gettimeofday(&t, 0);
    return t.tv_sec * 1e6 + t.tv_usec;
}
