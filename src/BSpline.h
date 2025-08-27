/******************************************************************************

  (c) 2005-2016 Scientific Computation Research Center,
      Rensselaer Polytechnic Institute. All rights reserved.

  This work is open source software, licensed under the terms of the
  BSD license as described in the LICENSE file in the top-level directory.

*******************************************************************************/
#ifndef BSPLINE_H
#define BSPLINE_H
#include <vector>
namespace Spline {

/**
Base class of 2D parametric curve
given parametric coordinate x, it evaluates
a. physical coordinate
b. first deriviative and normal vector
c. second deriviative and curvatur
*/
class Expression {
public:
  virtual double eval(double x) const = 0;
  virtual double evalFirstDeriv(double x) const = 0;
  virtual double evalSecondDeriv(double x) const = 0;
};

void dummyAnalyticExpression(double phi, double dummy, double *xyz,
                             void *userdata);
void evalCoord(double para, double *xyz, void *userdata);
void evalNormalVector(Expression *xp, Expression *yp, double para,
                      double *normalvec);
void evalCurvature(Expression *xp, Expression *yp, double para, double *curv);
int calcuBinomial(int j, int i);

/** Implementation of monomic polynomial*/
class PolyNomial : public Expression {
public:
  explicit PolyNomial(int degree_p, std::vector<double> &coffs_p);
  ~PolyNomial(){};
  void getcoeffs(std::vector<double> &coffs_p) const;
  virtual double eval(double x) const;
  virtual double evalFirstDeriv(double x) const;
  virtual double evalSecondDeriv(double x) const;
  void print();

private:
  int degree;
  std::vector<double> coffs;
  std::vector<double> firstDerivCoffs;
  std::vector<double> secondDerivCoffs;
};

// for clamped b-spline, the same as sim geo advanced
/** BSpline implementation
It also includes conversions between BSpline and monic polynomial
*/
class BSpline : public Expression {
public:
  BSpline(int order_p, std::vector<double> &ctrlPts_p,
          std::vector<double> &knots_p, std::vector<double> &weight_p);
  BSpline() : order(-1) {}
  ~BSpline(){};
  virtual double eval(double x) const;
  virtual double evalFirstDeriv(double x) const;
  virtual double evalSecondDeriv(double x) const;
  void print();
  void getpara(int &order_p, std::vector<double> &ctrlPts_p,
               std::vector<double> &knots_p, std::vector<double> &weight_p);
  BSpline &operator=(const PolyNomial &pn);

private:
  void calcuDerivCoeff();
  int order;
  std::vector<double> ctrlPts;
  std::vector<double> knots;
  std::vector<double> weight;

  std::vector<double> ctrlPts_1st;
  std::vector<double> ctrlPts_2nd;
};
}; // namespace Spline
#endif
