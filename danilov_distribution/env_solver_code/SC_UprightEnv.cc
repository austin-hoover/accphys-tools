/////////////////////////////////////////////////////////////////////////////
//
// FILE NAME
//   SC_UprightEnv.cc
//
// AUTHOR
//   Jeff Holmes
//   ORNL, jzh@ornl.gov
//
// CREATED
//   07/2019
//
// DESCRIPTION
//   File for calculating envelope evolution and space charge kicks
//   for upright uniform elliptical charge distributions including
//   a conducting wall boundary for space charge kicks
//
// REVISION HISTORY
//
/////////////////////////////////////////////////////////////////////////////


/////////////////////////////////////////////////////////////////////////////
//
// include files
//
/////////////////////////////////////////////////////////////////////////////

#include <iostream>

#include "SC_UprightEnv.h"


///////////////////////////////////////////////////////////////////////////
//
// NON-INLINE PROTECTED MEMBER FUNCTIONS FOR CLASS UprightEnvCalculator
//
///////////////////////////////////////////////////////////////////////////

Define_Standard_Members(UprightEnvCalculator, Object);

///////////////////////////////////////////////////////////////////////////
//
// CLASS NAME
//   UprightEnvCalculator
//
// INHERITANCE RELATIONSHIPS
//   UprightEnvCalculator -> Object
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

//-------------------------------------------------------------------------
void UprightEnvCalculator::_KickEnvelope(MacroPart& mp,
                                         double Intensity,
                                         double lkick)
{
  double SCTerm, XETerm, YETerm, afac;

  _doBoundary = 0;

  _a = mp._x(1);
  _b = mp._y(1);

  XETerm = _Emitx * _Emitx / (_a * _a * _a) * lkick;
  YETerm = _Emity * _Emity / (_b * _b * _b) * lkick;
  afac   = _a * _a + (2.0 * _Etax * _sigmaE) * (2.0 * _Etax * _sigmaE);
  SCTerm = 2.0 * Intensity * lkick/ (pow(afac, 0.5) + _b);

  mp._xp(1) += (XETerm + SCTerm);
  mp._yp(1) += (YETerm + SCTerm);

  _xsig2 = mp._x(1) * mp._x(1) / 4.0;
  _ysig2 = mp._y(1) * mp._y(1) / 4.0;
  
  if(_ysig2 > _xsig2)
  {
    _a = mp._y(1);
    _b = mp._x(1);
  }
  _asq = _a * _a;
  _bsq = _b * _b;
  _csq = (_a + _b) * (_a - _b);
  _c   = pow(_csq, 0.5);
  _circle = 0;
  if(_csq / (_asq + _bsq) < 1.0e-8) _circle = 1;
}

//-------------------------------------------------------------------------
double UprightEnvCalculator::_getu(double x, double y)
{
  double um, up, du, u, difm, difp, dif, dmp, tol;
  double xsq, ysq, snh, csh, snhsq, cshsq;
  du = 1.0;
  tol = 1.0e-12;
  xsq = x * x;
  ysq = y * y;

  if((xsq / _asq + ysq / _bsq) < 1.0)
  {
    return 0.0;
  }

  up = 0.0;
  snh = sinh(up);
  csh = cosh(up);
  snhsq = snh * snh;
  cshsq = csh * csh;
  difp = xsq * snhsq + ysq * cshsq - _csq * snhsq * cshsq;

  dmp = 1.0;
  while(dmp > 0.0)
  {
    um = up;
    difm = difp;
    up = um + du;
    snh = sinh(up);
    csh = cosh(up);
    snhsq = snh * snh;
    cshsq = csh * csh;
    difp = xsq * snhsq + ysq * cshsq - _csq * snhsq * cshsq;
    dmp = difm * difp;
  }
  while(du > tol)
  {
    u = 0.5 * (um + up);
    snh = sinh(u);
    csh = cosh(u);
    snhsq = snh * snh;
    cshsq = csh * csh;
    dif = xsq * snhsq + ysq * cshsq - _csq * snhsq * cshsq;
    dmp = difm * dif;
    if(dmp > 0.0)
    {
      um = u;
      difm = dif;
    }
    else
    {
      up = u;
      difp = dif;
    }
    du = up - um;
  }
  return u;
}

//-------------------------------------------------------------------------
double UprightEnvCalculator::_gettheta(double x, double y)
{
  double thm, thp, dth, th, difm, difp, dif, dmp, tol;
  double xsq, ysq, sn, cs, snsq, cssq;
  int iquit;

  double pi = 4.0 * atan2(1.0, 1.0);
  dth = pi / 20.0;
  tol = 1.0e-12;
  xsq = x * x;
  ysq = y * y;

  if((xsq / _asq + ysq / _bsq) < 1.0)
  {
    return 0.0;
  }

  thp = 0.0;
  sn = sin(thp);
  cs = cos(thp);
  snsq = sn * sn;
  cssq = cs * cs;
  difp = xsq * snsq - ysq * cssq - _csq * snsq * cssq;

  iquit = 0;
  dmp = 1.0;
  while(iquit == 0)
  {
    thm = thp;
    difm = difp;
    thp = thm + dth;
    sn = sin(thp);
    cs = cos(thp);
    snsq = sn * sn;
    cssq = cs * cs;
    difp = xsq * snsq - ysq * cssq - _csq * snsq * cssq;
    dmp = difm * difp;
    if(dmp < 0) iquit = 1;
    if(fabs(difp) < 1.e-08) iquit = 2;
  }
  if(iquit = 1)
  {
    while(dth > tol)
    {
      th = 0.5 * (thm + thp);
      sn = sin(th);
      cs = cos(th);
      snsq = sn * sn;
      cssq = cs * cs;
      dif = xsq * snsq - ysq * cssq - _csq * snsq * cssq;
      dmp = difm * dif;
      if(dmp > 0.0)
      {
        thm = th;
        difm = dif;
      }
      else
      {
        thp = th;
        difp = dif;
      }
      dth = thp - thm;
    }
  }
  else if(iquit = 2)
  {
    th = thp;
  }
  if(x < 0.0)
  {
    if(y < 0.0)
    {
      th = pi + th;
    }
    else
    {
      th = pi - th;
    }
  }
  else
  {
    if(y < 0.0)
    {
      th = - th;
    }
  }
  return th;
}

//-------------------------------------------------------------------------
void UprightEnvCalculator::_BoundaryCoeffs()
{
  int i, m;
  double x, y, u, th, rsq;

  double* BPPhi;
  BPPhi = new double[_BPPoints];
  for(i = 0; i < _BPPoints; i++)
  {
    BPPhi[i] = 0.0;
    if(_ysig2 > _xsig2)
    {
      x =  _BPy[i];
      y = -_BPx[i];
    }
    else
    {
      x =  _BPx[i];
      y =  _BPy[i];
    }

    if((x * x / _asq + y * y / _bsq) < 1.0)
    {
      cout << "Beam Crosses Boundary, _asq, _bsq, _BPx, _BPy = "
           << _asq    << "   " << _bsq    << "   "
           << _BPx[i] << "   " << _BPy[i] << "\n";
      return;
    }
    if(_circle == 0)
    {
      u  = _getu(x, y);
      th = _gettheta(x, y);
      BPPhi[i] = -u - 0.5 * exp(-2.0 * u) * cos(2.0 * th);
    }
    else
    {
      rsq = x * x + y * y;
      BPPhi[i] = -0.5 * log(rsq);
    }
  }

  double* RHS;
  RHS = new double[2 * _BPModes + 1];
  for(m = 0; m < 2 * _BPModes + 1; m++)
  {
    RHS[m] = 0.0;
    for(i = 0; i < _BPPoints; i++)
    {
      RHS[m] -= _BPPhiH1[i][m] * BPPhi[i];
    }
  }

  for(m = 0; m < 2 * _BPModes + 1; m++)
  {
    _BPCoeffs[m] = 0.0;
    for(i = 0; i < 2 * _BPModes + 1; i++)
    {
      _BPCoeffs[m] += _BPPhiH2[m][i] * RHS[i];
    }
  }

  double BPresi, BPresf;
  BPresi = 0.0;
  BPresf = 0.0;
  for(i = 0; i < _BPPoints; i++)
  {
    BPresi += BPPhi[i] * BPPhi[i];
    for(m = 0; m < 2 * _BPModes + 1; m++)
    {
      BPPhi[i] += _BPPhiH1[i][m] * _BPCoeffs[m];
    }
    BPresf += BPPhi[i] * BPPhi[i];
  }

/*
  double BPrat = 1.0;
  if(BPresi != 0.0) BPrat = BPresf / BPresi;
  cout << "Boundary residual BPresi, BPresf, BPrat = "
       << BPresi << "   " << BPresf << "   " << BPrat << "\n";
*/

  delete BPPhi;
  delete RHS;

  _doBoundary = 1;
}

//-------------------------------------------------------------------------
void UprightEnvCalculator::_ApplyForce(MacroPart& mp,
                                       double Intensity,
                                       double lkick)
{
  int i;
  double x, y, u, th;
  double apb, xapb, yapb;
  double expp, expm, csh, snh, cs, sn;
  double exp2, cs2, sn2, dudx, dudy, dthdx, dthdy, denom;

  int m, mc, ms;
  double r, rmc, rms, rsq, xcoeff, ycoeff;
  std::complex<double> z;
  std::complex<double> zm;

  double kickx, kicky, kickbndx, kickbndy, kickfactor;

  kickfactor = Intensity * lkick;

  apb  = _a + _b;
  xapb = 2.0 / (_a * apb);
  yapb = 2.0 / (_b * apb);

  for(i = 2; i <= mp._nMacros; i++)
  {
    if(_ysig2 > _xsig2)
    {
      x =  mp._y(i);
      y = -mp._x(i);
    }
    else
    {
      x =  mp._x(i);
      y =  mp._y(i);
    }


    if((x * x / _asq + y * y / _bsq) > 1.0)
    {
      if(_circle == 0)
      {
        u     = _getu(x, y);
        th    = _gettheta(x, y);
        expp  = exp(u);
        expm  = 1.0 / expp;
        csh   = 0.5 * (expp + expm);
        snh   = 0.5 * (expp - expm);
        cs    = cos(th);
        sn    = sin(th);
        exp2  = expm * expm;
        cs2   = cs * cs - sn * sn;
        sn2   = 2.0 * cs * sn;
        denom = _c * (snh * snh + sn * sn);
        dudx  =  snh * cs / denom;
        dudy  =  csh * sn / denom;
        dthdx = -csh * sn / denom;
        dthdy =  snh * cs / denom;
        kickx = (1.0 - exp2 * cs2) * dudx -exp2 * sn2 * dthdx;
        kicky = (1.0 - exp2 * cs2) * dudy -exp2 * sn2 * dthdy;
      }
      else
      {
        rsq = x * x + y * y;
        kickx = x / rsq;
	kicky = y / rsq;
      }
    }
    else
    {
      kickx = xapb * x;
      kicky = yapb * y;
    }

    kickbndx = 0.0;
    kickbndy = 0.0;
    if(_doBoundary != 0)
    {
      x =  mp._x(i);
      y =  mp._y(i);
      rsq = x * x + y * y;
      xcoeff = x / rsq;
      ycoeff = y / rsq;
      z = std::complex<double>(x / _BPrnorm, y / _BPrnorm);
      zm = std::complex<double>(1.0, 0.0);
      for(m = 1; m <= _BPModes; m++)
      {
        zm  = z * zm;
        rmc = std::real(zm);
        rms = std::imag(zm);
        mc = 2 * m - 1;
        ms = 2 * m;
        kickbndx -= double(m) *
                   (_BPCoeffs[mc] * (rmc * xcoeff + rms * ycoeff) +
                    _BPCoeffs[ms] * (rms * xcoeff - rmc * ycoeff));
        kickbndy -= double(m) *
                   (_BPCoeffs[mc] * (rmc * ycoeff - rms * xcoeff) +
                    _BPCoeffs[ms] * (rms * ycoeff + rmc * xcoeff));
      }
    }

    if(_ysig2 > _xsig2)
    {
      mp._xp(i) += -kickfactor * (kicky - kickbndx);
      mp._yp(i) +=  kickfactor * (kickx + kickbndy);
    }
    else
    {
      mp._xp(i) +=  kickfactor * (kickx + kickbndx);
      mp._yp(i) +=  kickfactor * (kicky + kickbndy);
    }
  }
}

//-------------------------------------------------------------------------
void UprightEnvCalculator::_Clear()
{
  _doBoundary = 0;
}
