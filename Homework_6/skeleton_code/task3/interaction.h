#pragma once

#include <cmath>

struct Interaction {
public:
    Interaction(double cutoff, double alpha) :
        cutoffSqr_{cutoff * cutoff}, alpha_{alpha}
    { }

    /// Compute the force that q exerts on p.
    __device__ double2 operator()(double2 p, double2 q) {
        // Sign such that the force is repulsive.
        const double dx = p.x - q.x;
        const double dy = p.y - q.y;
        const double rr = dx * dx + dy * dy;
        if (rr > cutoffSqr_)
            return double2{};
        const double t = rsqrt(alpha_ + rr);
        const double inv = t * t * t * (t * t);
        double2 f;
        f.x = dx * inv;
        f.y = dy * inv;
        return f;
    }

    double getAlpha() const {
        return alpha_;
    }

    double getCutoff() const {
        return std::sqrt(cutoffSqr_);
    }

private:
    double cutoffSqr_;
    double alpha_;
};


struct CellListInfo;

/// Compute the total force on each of the particles.
void computeForces(
        CellListInfo info,
        Interaction interaction,
        const double2 *pSortedDev,
        double2 *f,
        int numParticles);
