#include "runge_kutta.hh"

#include <iostream>

struct QO {
      QO(double x0, double v0, double lambda) : _x0(x0), _v0(v0), _lambda(lambda) {};

      void clone(const QO &other)
      {
         _x0 = other._x0;
         _v0 = other._v0;
      }

      void add_with_weight(double w, const QO &other)
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
         return _v0 * _v0 * 0.5 + _x0 * _x0 * 0.5 + _lambda * pow(_x0, 4.) * 0.25;
      }

      double _x0, _v0, _lambda;
};

int main()
{
   QO qo(1.0, 0., 1.0);
   std::vector<double> time_points;
   for (size_t i = 0; i < 100; i++) {
      time_points.push_back(i);
   }
   TimeInfo ti(time_points, 0.1);

   auto tableau = PreImplementedTableau::DOPRI8;

   rk_rhs_t<QO> QO_rhs = [](double, QO &in) -> void {
      double v = in._v0;
      in._v0   = -in._x0 - in._lambda * pow(in._x0, 3.);
      in._x0   = v;
   };

   std::function<double(double)> pert_sol_l1 = [&](double t) -> double {
      // Note: this is bound to the initial conditions of x0 = 1, v0 = 0
      return cos(t) - qo._lambda * 3. * t * sin(t) / 8. - qo._lambda * sin(t) * sin(2.0 * t) / 16.0;
   };

   std::function<double(double)> pert_sol_l2 = [&](double t) -> double {
      // Note: this is bound to the initial conditions of x0 = 1, v0 = 0
      return pert_sol_l1(t) + qo._lambda * qo._lambda *
                                  ((23 * cos(t)) / 1024. - (9 * pow(t, 2) * cos(t)) / 128. -
                                   (3 * cos(3 * t)) / 128. + (5 * cos(5 * t)) / 1024. +
                                   (3 * t * sin(t)) / 32. - (9 * t * sin(3 * t)) / 256.);
   };

   std::function<double(double)> resum_sol_l1 = [&](double t) -> double {
      return cos(t + (qo._lambda / 8.) * (3 * t + cos(t) * sin(t)));
   };

   rk_callback_t<QO> callback = [&](double t, const TimeInfo &, const QO &s) {
      std::cout << t << " " << s._x0;
      std::cout << " " << pert_sol_l2(t);
      std::cout << " " << resum_sol_l1(t) << std::endl;
   };

   rk_callback_t<QO> callback_e = [&](double t, const TimeInfo &, const QO &s) {
      std::printf("%.4f\t%.10e\n", t, s.energy());
   };

   auto solver = RungeKutta<QO, tableau.order>(tableau, qo, ti, QO_rhs);
   solver.AddCallback(callback_e);

   solver();

   // std::cout << solver.GetSolution()._x0 << std::endl;
}