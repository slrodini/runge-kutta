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

      double energy() const
      {
         return _v0 * _v0 * 0.5 + _x0 * _x0 * 0.5;
      }

      double _x0, _v0;
};

int main()
{
   HO ho(1.0, 0.);
   std::vector<double> time_points;
   for (size_t i = 0; i < 100; i++) {
      time_points.push_back(i);
   }
   TimeInfo ti(time_points, 0.1);

   auto tableau = PreImplementedTableau::DOPRI8;

   rk_rhs_t<HO> ho_rhs = [](double, HO &in) -> void {
      double v = in._v0;
      in._v0   = -in._x0;
      in._x0   = v;
      // in._x0 = -in._x0;
   };

   rk_callback_t<HO> callback = [](double t, const TimeInfo &, const HO &s) {
      std::cout << t << " " << s._x0 << std::endl;
   };

   rk_callback_t<HO> callback_e = [&](double t, const TimeInfo &, const HO &s) {
      std::printf("%.4f\t%.10e\n", t, s.energy());
   };


   auto solver = RungeKutta<HO, tableau.order>(tableau, ho, ti, ho_rhs);
   solver.AddCallback(callback_e);

   solver();
}