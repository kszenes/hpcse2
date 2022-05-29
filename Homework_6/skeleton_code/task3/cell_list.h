#pragma once

#include "scan.h"

struct Interaction;

/// Cell list metadata used from various kernels.
/// (useful both in cell lists building and in force computation)
struct CellListInfo {
    /// Inverse cell size.
    double2 invCellSize;

    /// Number of cells for each dimension.
    int2 numCells;

    /// Pointer to the offsets array.
    int *offsets;

    // TODO: (optional) Add here any utility functions here you might find
    // useful, such as computing cell index from coordinates.

    __device__ int2 computeIndices(double2 coord) const {
        int cx = (int) coord.x * invCellSize.x;
        int cy = (int) coord.y * invCellSize.y;
        return {cx, cy};
    }

    __device__ int computeIndex(double2 coord) const {
        int2 c = computeIndices(coord);
        return c.y * numCells.x + c.x;
    }
};

class CellList {
public:
    /// Create cell list that spans across the given domain with given cell size.
    CellList(double2 domainSize, double cellSize);
    ~CellList();

    /// Build cell list for given particles and reorder the particle positions.
    void build(const double2 *pDev, double2 *pSortedDev, int numParticles);

    /// Return the cell list info / metadata.
    CellListInfo getInfo() const {
        return {invCellSize_, numCells_, offsetsDev_};
    }
private:
    Scan scan_;

    double2 invCellSize_;
    int2 numCells_;
    int *countsDev_ = nullptr;
    int *offsetsDev_ = nullptr;
};
