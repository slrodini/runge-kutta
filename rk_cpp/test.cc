#include "runge_kutta.hh"

#include <iostream>

struct HO {
      HO(double x0, double v0) : _x0(x0), _v0(v0) {};

      void clone(const HO &other)
      {
         _x0 = other._x0;
         _v0 = other._v0;
      }

      void add_with_weight(double w, const HO &other)
      {
         _x0 += w * other._x0;
         _v0 += w * other._v0;
      }

      void scalar_mult(double s)
      {
         _x0 *= s;
         _v0 *= s;
      }

      double _x0, _v0;
};

int main()
{
   HO ho(1.0, 0.);
   TimeInfo ti({0., 10.}, 0.5);

   auto tableau = PreImplementedTableau::DOPRI8;
   // auto tableau = PreImplementedTableau::RKOriginal;

   rk_rhs_t<HO> ho_rhs = [](double t, HO &in) -> void {
      double v = in._v0;
      in._v0   = -in._x0;
      in._x0   = v;
      // in._x0 = -in._x0;
   };

   rk_callback_t<HO> callback = [](double t, const TimeInfo &, const HO &s) {
      std::cout << t << " " << s._x0 << std::endl;
   };

   auto solver = RungeKutta<HO, tableau.order>(tableau, ho, ti, ho_rhs);
   solver.AddCallback(callback);

   solver();

   std::cout << solver.GetSolution()._x0 << std::endl;
}