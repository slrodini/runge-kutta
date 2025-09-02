#include "runge_kutta.hh"

#include <iostream>

using namespace rk;

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

      void make_zero()
      {
         _x0 = 0;
         _v0 = 0;
      }

      double norm()
      {
         // return std::max(std::fabs(_x0), std::fabs(_v0));
         return sqrt(_x0*_x0 + _v0*_v0);
      }

      double _x0, _v0;
};

int main2()
{
   vd<1> x0({1.});
   vd<1> v0({0.});
   auto tableau = PreImplementedTableau::OS76;
   std::vector<double> time_points;
   for (size_t i = 0; i < 100; i++) {
      time_points.push_back(i);
   }
   TimeInfo ti(time_points, 0.1);

   rkn_rhs_t<1> ho_rhs = [](double, vd<1> &v)
   {
      v[0] = -v[0];
   };

   rkn_callback_t<1> callback = [](double t, const TimeInfo &, const vd<1> &s, const vd<1> &v) {
      double energy = 0.5 * (s(0) * s(0) + v(0) * v(0));
      std::printf("%.4f\t%.10e\t%.10e\n", t, s(0) - cos(t),energy);

   };
   auto solver = RungeKuttaNystrom<tableau.stages, 1>(tableau, x0, v0, ti, ho_rhs, 1.0e-10);
   solver.AddCallback(callback);

   solver();

   return 0;
}

int main1()
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
      std::cout << t << " " << s._x0  - cos(t) << std::endl;
   };

   rk_callback_t<HO> callback_e = [&](double t, const TimeInfo &, const HO &s) {
      std::printf("%.4f\t%.10e\n", t, s.energy());
   };


   auto solver = RungeKutta<HO, tableau.stages>(tableau, ho, ti, ho_rhs);
   solver.AddCallback(callback);
   solver.CallbackEachStep();

   solver();

   return 0;
}


int main3()
{
   HO ho(1.0, 0.);
   std::vector<double> time_points;
   for (size_t i = 0; i < 10; i++) {
      time_points.push_back(10*i);
   }
   TimeInfo ti(time_points, 6.);

   auto tableau = PreImplementedTableau::DOPRI54;

   rk_rhs_t<HO> ho_rhs = [](double, HO &in) -> void {
      double v = in._v0;
      in._v0   = -in._x0;
      in._x0   = v;
   };

   rk_callback_t<HO> callback = [](double t, const TimeInfo &, const HO &s) {
      std::cout << t << " " << s._x0  - cos(t) << std::endl;
   };

   rk_callback_t<HO> callback_e = [&](double t, const TimeInfo &, const HO &s) {
      std::printf("%.4f\t%.10e\n", t, s.energy());
   };


   auto solver = AdaptiveRungeKutta<HO, tableau.stages>(tableau, ho, ti, ho_rhs, 1.0e-10);
   // auto solver = RungeKutta<HO, tableau.stages>(tableau, ho, ti, ho_rhs);

   solver.AddCallback(callback);
   // solver.CallbackEachStep();

   solver();

   return 0;
}

int main()
{
   return main2();
}