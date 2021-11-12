#if !defined(__SCUprightEnvelope__)
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
//   UprightEnvCalculator
//
// INHERITANCE RELATIONSHIPS
//   None
//
// USING/CONTAINING RELATIONSHIPS
//   None
//
// DESCRIPTION
//   Class for calculating envelope evolution and space charge kicks
//   for upright uniform elliptical charge distributions including
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

class UprightEnvCalculator : public Object
{
  Declare_Standard_Members(UprightEnvCalculator, Object);

  public:

  // Constructor

  UprightEnvCalculator(double Emitx, double Emity,
                       double Etax, double sigmaE,
                       int BPPoints, int BPModes,
                       double BPrnorm, double BPResid,
                       double* BPx , double* BPy,
                       double** BPPhiH1, double** BPPhiH2) :
		       _Emitx(Emitx), _Emity(Emity),
		       _Etax(Etax), _sigmaE(sigmaE),
                       _BPPoints(BPPoints), _BPModes(BPModes),
                       _BPrnorm(BPrnorm), _BPResid(BPResid),
                       _BPx(BPx), _BPy(BPy),
                       _BPPhiH1(BPPhiH1), _BPPhiH2(BPPhiH2)
  {
    _doBoundary = 0;
    _BPCoeffs = new double[2 * _BPModes + 1];
  }

  // Destructor

  ~UprightEnvCalculator()
  {
    delete _BPCoeffs;
  }

  // Variables and Methods

  double _Emitx;
  double _Emity;
  double _Etax;
  double _sigmaE;
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

  double _xsig2;
  double _ysig2;
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

  void _Clear();
};


#endif   // __SCUprightEnvelope__
