#include "Equation2D.h"

class WaveEquation:public Equation2D
{
  /*
   * Derived class from Equation2D. Solves the 2D wave equation.
   */
  public:
    WaveEquation(const int a_N, const MPI_Comm a_comm):Equation2D(a_N,a_comm){};

  protected:
    virtual double initialCondition(const double x, const double y) override;

    virtual void applyStencil(const int i, const int j) override;

    virtual void computeTimestep() override;

    virtual void derivedFunctionCalls() override;

  private:
    double c_aug;         // auxiliary quantity
    double energy_norm_m; // auxiliary quantities used to compute energy norm
    double energy_norm_a; // auxiliary quantities used to compute energy norm
};