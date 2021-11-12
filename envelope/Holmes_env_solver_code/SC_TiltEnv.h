#if !defined(__SCTiltEnvelope__)
#define __SCEnvelope__

#include <cstdlib>
#include <cstdio>
#include <cmath>
#include <mpi.h>
#include <vector>
#include <complex>
#include "Object.h"
#include "MacroPart.h"

using namespace std;


///////////////////////////////////////////////////////////////////////////
//
// CLASS NAME
//   TiltEnvCalculator
//
// INHERITANCE RELATIONSHIPS
//   None
//
// USING/CONTAINING RELATIONSHIPS
//   None
//
// DESCRIPTION
//   Class for calculating envelope evolution and space charge kicks
//   for tilted uniform elliptical charge distributions including
//   a conducting wall boundary for space charge kicks
//
// PUBLIC MEMBERS
//
// PROTECTED MEMBERS
//   None
//
// PRIVATE MEMBERS
//   None
//
///////////////////////////////////////////////////////////////////////////

class TiltEnvCalculator : public Object
{
  Declare_Standard_Members(TiltEnvCalculator, Object);

  public:

  // Constructor

  TiltEnvCalculator(int BPPoints, int BPModes,
                    double BPrnorm, double BPResid,
                    double* BPx , double* BPy,
                    double** BPPhiH1, double** BPPhiH2) :
                    _BPPoints(BPPoints), _BPModes(BPModes),
                    _BPrnorm(BPrnorm), _BPResid(BPResid),
                    _BPx(BPx), _BPy(BPy),
                    _BPPhiH1(BPPhiH1), _BPPhiH2(BPPhiH2)
  {
    _doBoundary = 0;
    _BPCoeffs = new double[2 * _BPModes + 1];
  }

  // Destructor

  ~TiltEnvCalculator()
  {
    delete _BPCoeffs;
  }

  // Variables and Methods

  int _doBoundary;
  int _BPPoints;
  int _BPModes;
  double _BPrnorm;
  double _BPResid;
  double* _BPx;
  double* _BPy;
  double** _BPPhiH1;
  double** _BPPhiH2;
  double* _BPCoeffs;

  double _xc;
  double _yc;
  double _xs;
  double _ys;
  double _tilt;
  double _cst;
  double _snt;
  double _cxsq;
  double _cysq;
  double _a;
  double _b;
  double _c;
  double _asq;
  double _bsq;
  double _csq;

  int _circle;

  void _KickEnvelope(MacroPart& mp, double Intensity, double lkick);

  double _getu(double x, double y);
  double _gettheta(double x, double y);

  void _BoundaryCoeffs();

  void _ApplyForce(MacroPart& mp, double Intensity, double lkick);

  void _getEllipsDat(double xc, double yc, double xs, double ys,
                     double& tilt, double& cxsq, double& cysq);
  void _Clear();
};


#endif   // __SCTiltEnvelope__
