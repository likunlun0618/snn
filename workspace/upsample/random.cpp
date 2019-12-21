#include <cstdlib>
#include <ctime>
#include "random.h"

void seed()
{
    srand(time(NULL));
}

void seed(int s)
{
    srand(s);
}

float randf()
{
    return (float)rand() / (float)RAND_MAX;
}
