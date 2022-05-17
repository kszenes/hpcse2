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
    void *bufferDev_ = nullptr;
    size_t bufferSize_ = 0;
};
