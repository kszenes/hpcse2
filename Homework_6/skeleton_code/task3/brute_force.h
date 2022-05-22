#pragma once

#include "interaction.h"

/// Compute forces using an N^2 algorithm (O(N) per thread).
void computeForcesSlow(
        Interaction interaction,
        const double2 *p,
        double2 *f,
        int numParticles);
