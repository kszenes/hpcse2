#include "cell_list.h"
#include "interaction.h"
#include "../include/utils.h"

CellList::CellList(double2 domainSize, double cellSize) {
    numCells_.x = (int)std::ceil(domainSize.x / cellSize);
    numCells_.y = (int)std::ceil(domainSize.y / cellSize);
    invCellSize_.x = numCells_.x / domainSize.x;
    invCellSize_.y = numCells_.y / domainSize.y;

    // TODO: Allocate buffers. Note that offsets array needs 1 element more
    // than the counts array!
}

CellList::~CellList() {
    // TODO: Deallocate buffers.
}

void CellList::build(const double2 *pDev, double2 *pSortedDev, int numParticles) {
    // Do not complain about unused variable. Erase once the code has been implemented.
    (void)pDev;
    (void)pSortedDev;
    (void)numParticles;

    // const CellListInfo info = getInfo();

    // Stage 1: compute cell sizes
    // TODO: Compute cell sizes (number of particles per cell).

    // Stage 2: compute offsets
    // TODO: Compute offsets using the Scan class (use the scan_ member variable).

    // Stage 3: reorder particles into cells
    // TODO: Reorder the particles in the sorted order and save them to pSortedDev.
}
