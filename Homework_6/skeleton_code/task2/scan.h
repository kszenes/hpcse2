#pragma once

#include <cstddef>

class Scan {
public:
    Scan();
    ~Scan();

    /** Compute the inclusive sum.

        For example, for an input array
            {5, 3,  5,  1,  2,  0,  2}
        compute
            {5, 8, 13, 14, 16, 16, 18}.
    */
    void inclusiveSum(const int *inDev, int *outDev, int N);

private:
    // TODO: Put here pointers to any temporary buffers you might need, such
    // that no allocation occurs if the `inclusiveSum` function is invoked
    // multiple times with the same `N` (apart from the first run, of course).
    // In practice, avoiding allocation/deallocation all the time is desirable
    // because cudaFree must synchronize the device!
    int* tmp;
};
