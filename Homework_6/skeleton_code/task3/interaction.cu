#include "cell_list.h"
#include "interaction.h"

void computeForces(
        const CellListInfo info,
        Interaction interaction,
        const double2 *pSortedDev,
        double2 *f,
        int numParticles) {

    // Do not complain about unused variable. Erase once the code has been implemented.
    (void)info;
    (void)interaction;
    (void)pSortedDev;
    (void)f;
    (void)numParticles;

    // TODO: Implement the optimized version of computeForcesSlow (see
    // brute_force.cu) that utilized cell lists.
}
