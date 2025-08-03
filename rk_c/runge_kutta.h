#ifndef RUNGE_KUTTA_H
#define RUNGE_KUTTA_H
#include <stdio.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <stdbool.h>

typedef struct {
      size_t order;
      double *_ci, *_bi;
      double **_aij;
} ButcherTableau;

void rk_print_tableau(const ButcherTableau *t);

void rk_copy_tableau(ButcherTableau *dest, const ButcherTableau *src);

void rk_free_tableau(ButcherTableau *t);

typedef struct {
      void *data;
      void (*init_data)(void **data);
      void (*free_data)(void **data);
      void (*clone)(void *dest, void *src);
      void (*add_with_weight)(void *dest, double w, void *src);
      void (*scalar_mult)(void *dest, double s);
} Solution;

void rk_copy_solution(Solution *dest, Solution *src);
void rk_free_solution(Solution *s);

typedef struct {
      size_t size;
      double *_ts, *_dt;
      size_t *_n_step;
} TimeInfo;

TimeInfo rk_generate_time_info(double time_stamps[], size_t l, double dt_min, size_t n_step_min);
void rk_free_time_info(TimeInfo *t);

typedef void (*rk_rhs_t)(double, Solution *);
typedef void (*rk_callback_t)(double, const TimeInfo *const, Solution *);

typedef struct {
      ButcherTableau tableau;
      Solution solution, _temp_step;
      Solution *_ki;
      double _t, _dt;
      TimeInfo _time_info;
      rk_rhs_t _rhs;
      rk_callback_t _callback;
      bool _cb_each_step;
} RungeKuttaContext;

typedef enum { RKOriginal_T = 0, DOPRI5_T = 1, DOPRI8_T = 2 } PreImplementedTableau;

void rk_free_runge_kutta_context(RungeKuttaContext **c);
RungeKuttaContext *rk_init_context_from_tab(const ButcherTableau *tab, Solution *initial_condition,
                                            TimeInfo time_info, rk_rhs_t rhs,
                                            rk_callback_t callback);

RungeKuttaContext *rk_init_context(PreImplementedTableau pit, Solution *initial_condition,
                                   TimeInfo time_info, rk_rhs_t rhs, rk_callback_t callback);
void rk_step(RungeKuttaContext *context);

void rk_evolve(RungeKuttaContext *context);

#ifdef RK_IMPLEMENTATION
#include <math.h>

static const ButcherTableau RKOriginal = {
    .order = 4,
    ._ci   = (double[4]){0., 0.5, 0.5, 1.0},
    ._bi   = (double[4]){1. / 6., 1. / 3., 1. / 3., 1. / 6.},
    ._aij =
        (double *[4]){
            NULL,
            (double[]){0.5},
            (double[]){0., 0.5},
            (double[]){0., 0., 1.},
        },
};

static const ButcherTableau DOPRI5 = {
    .order = 7,
    ._ci   = (double[7]){0., 1. / 5, 3. / 10, 4. / 5, 8. / 9, 1., 1.},
    ._bi   = (double[7]){35. / 384, 0., 500. / 1113, 125. / 192, -2187. / 6784, 11. / 84, 0.},
    ._aij =
        (double *[7]){
            NULL,
            (double[]){1. / 5},
            (double[]){3. / 40, 9. / 40},
            (double[]){44. / 45, -56. / 15, 32. / 9},
            (double[]){19372. / 6561, -25360. / 2187, 64448. / 6561, -212. / 729},
            (double[]){9017. / 3168, -355. / 33, 46732. / 5247, 49. / 176, -5103. / 18656},
            (double[]){35. / 384, 0., 500. / 1113, 125. / 192, -2187. / 6784, 11. / 84},
        },
};

static const ButcherTableau DOPRI8 = {
    .order = 13,
    ._ci   = (double[13]){0., 1. / 18, 1. / 12, 1. / 8, 5. / 16, 3. / 8, 59. / 400, 93. / 200, 5490023248. / 9719169821, 13. / 20, 1201146811. / 1299019798, 1., 1.},
    ._bi   = (double[13]){14005451. / 335480064, 0., 0., 0., 0., -59238493. / 1068277825, 181606767. / 758867731, 561292985. / 797845732, -1041891430. / 1371343529, 760417239. / 1151165299, 118820643. / 751138087, -528747749. / 2220607170, 1. / 4},
    ._aij =
        (double *[13]){
            NULL,
            (double[]){1./18},
            (double[]){1./48, 1./16},
            (double[]){1./32, 0., 3./32},
            (double[]){5./16, 0., -75./64, 75./64},
            (double[]){3./80, 0., 0., 3./16, 3./20},
            (double[]){29443841./614563906, 0., 0., 77736538./692538347, -28693883./1125000000, 23124283./1800000000},
            (double[]){16016141./946692911, 0., 0., 61564180./158732637, 22789713./633445777, 545815736./2771057229, -180193667./1043307555},
            (double[]){39632708./573591083, 0., 0., -433636366./683701615, -421739975./2616292301, 100302831./723423059, 790204164./839813087, 800635310./3783071287},
            (double[]){246121993./1340847787, 0., 0., -37695042795./15268766246, -309121744./1061227803, -12992083./490766935, 6005943493./2108947869, 393006217./1396673457, 123872331./1001029789},
            (double[]){-1028468189./846180014, 0., 0., 8478235783./508512852, 1311729495./1432422823, -10304129995./1701304382, -48777925059./3047939560, 15336726248./1032824649, -45442868181./3398467696, 3065993473./597172653},
            (double[]){185892177./718116043, 0., 0., -3185094517./667107341, -477755414./1098053517, -703635378./230739211, 5731566787./1027545527, 5232866602./850066563, -4093664535./808688257, 3962137247./1805957418, 65686358./487910083},
            (double[]){403863854./491063109, 0., 0., -5068492393./434740067, -411421997./543043805, 652783627./914296604, 11173962825./925320556, -13158990841./6184727034, 3936647629./1978049680, -160528059./685178525, 248638103./1413531060, 0.}
        },
};

void rk_print_tableau(const ButcherTableau *t)
{
   printf("Order: %zu\n", t->order);

   printf("C[i] = ");
   for (size_t i = 0; i < t->order; i++) {
      printf("%.16f", t->_ci[i]);
      if (i != t->order - 1) {
         printf(", ");
      }
   }
   printf("\n");

   printf("B[i] = ");
   for (size_t i = 0; i < t->order; i++) {
      printf("%.16f", t->_bi[i]);
      if (i != t->order - 1) {
         printf(", ");
      }
   }
   printf("\n");

   printf("A[i][j] = ");
   for (size_t i = 0; i < t->order; i++) {
      for (size_t j = 0; j < i; j++) {
         printf("%.16f", t->_aij[i][j]);
         if (j != i - 1) {
            printf(", ");
         }
      }
      printf("\n          ");
   }
   printf("\n");
}

void rk_copy_tableau(ButcherTableau *dest, const ButcherTableau *src)
{
   size_t n    = src->order;
   dest->order = n;
   dest->_bi   = (double *)calloc(n, sizeof(double));
   dest->_ci   = (double *)calloc(n, sizeof(double));
   dest->_aij  = (double **)calloc(n, sizeof(double *));
   for (size_t i = 0; i < n; i++) {
      dest->_aij[i] = src->_aij[i] == NULL ? NULL : (double *)calloc(i, sizeof(double));
      dest->_bi[i]  = src->_bi[i];
      dest->_ci[i]  = src->_ci[i];
      for (size_t j = 0; j < i; j++) {
         dest->_aij[i][j] = src->_aij[i][j];
      }
   }
}

void rk_free_tableau(ButcherTableau *t)
{
   free(t->_bi);
   free(t->_ci);
   for (size_t i = 0; i < t->order; i++) {
      free(t->_aij[i]);
   }
   free(t->_aij);
}

void rk_copy_solution(Solution *dest, Solution *src)
{
   if (dest == NULL || src == NULL) exit(-1); // ToDo: Add error msg
   src->init_data(&(dest->data));
   src->clone(dest->data, src->data);
   dest->init_data       = src->init_data;
   dest->free_data       = src->free_data;
   dest->clone           = src->clone;
   dest->add_with_weight = src->add_with_weight;
   dest->scalar_mult     = src->scalar_mult;
}

void rk_free_solution(Solution *s)
{
   s->free_data(&(s->data));
}

TimeInfo rk_generate_time_info(double time_stamps[], size_t l, double dt_min, size_t n_step_min)
{
   if (l <= 1) exit(-1); // ToDo: add error msg
   // ToDo: sort time_stamps

   TimeInfo time_info = {
       .size    = l - 1,
       ._ts     = (double *)calloc(l, sizeof(double)),
       ._dt     = (double *)calloc(l - 1, sizeof(double)),
       ._n_step = (size_t *)calloc(l - 1, sizeof(size_t)),
   };

   double nd = (double)n_step_min;
   for (size_t i = 0; i < l - 1; i++) {
      time_info._ts[i] = time_stamps[i];
      size_t n_step    = n_step_min;

      double dt_tmp = (time_stamps[i + 1] - time_stamps[i]) / nd;
      while (dt_tmp > dt_min) {
         n_step *= 2;
         dt_tmp = (time_stamps[i + 1] - time_stamps[i]) / ((double)n_step);
      }
      time_info._dt[i]     = dt_tmp;
      time_info._n_step[i] = n_step;
   }

   time_info._ts[l - 1] = time_stamps[l - 1];

   return time_info;
}

void rk_free_time_info(TimeInfo *t)
{
   free(t->_ts);
   free(t->_dt);
   free(t->_n_step);
}

void rk_free_runge_kutta_context(RungeKuttaContext **c_d)
{
   RungeKuttaContext *c = *c_d;
   rk_free_solution(&c->solution);
   rk_free_solution(&c->_temp_step);
   for (size_t i = 0; i < c->tableau.order; i++) {
      rk_free_solution(&(c->_ki[i]));
   }
   free(c->_ki);

   rk_free_tableau(&c->tableau);
   rk_free_time_info(&c->_time_info);
   free(*c_d);
   *c_d = NULL;
}

RungeKuttaContext *rk_init_context_from_tab(const ButcherTableau *tab, Solution *initial_condition,
                                            TimeInfo time_info, rk_rhs_t rhs,
                                            rk_callback_t callback)
{
   if (rhs == NULL) return NULL;

   RungeKuttaContext *context = (RungeKuttaContext *)calloc(1, sizeof(RungeKuttaContext));

   context->_time_info.size    = time_info.size;
   context->_time_info._ts     = (double *)calloc(time_info.size + 1, sizeof(double));
   context->_time_info._dt     = (double *)calloc(time_info.size, sizeof(double));
   context->_time_info._n_step = (size_t *)calloc(time_info.size, sizeof(size_t));
   for (size_t i = 0; i < time_info.size; i++) {
      context->_time_info._ts[i]     = time_info._ts[i];
      context->_time_info._dt[i]     = time_info._dt[i];
      context->_time_info._n_step[i] = time_info._n_step[i];
   }
   context->_time_info._ts[time_info.size] = time_info._ts[time_info.size];

   rk_copy_solution(&context->solution, initial_condition);
   rk_copy_solution(&context->_temp_step, initial_condition);



   context->_ki = (Solution *)calloc(tab->order, sizeof(Solution));
   for (size_t i = 0; i < tab->order; i++) {
      rk_copy_solution(&(context->_ki[i]), initial_condition);
   }


   rk_copy_tableau(&context->tableau, tab);

   context->_rhs      = rhs;
   context->_callback = callback;

   context->_cb_each_step = false;

   return context;
}

RungeKuttaContext *rk_init_context(PreImplementedTableau pit, Solution *initial_condition,
                                   TimeInfo time_info, rk_rhs_t rhs, rk_callback_t callback)
{
   const ButcherTableau *tab = NULL;
   switch (pit) {
   case RKOriginal_T:
      tab = &RKOriginal;
      break;
   case DOPRI5_T:
      tab = &DOPRI5;
      break;
   case DOPRI8_T:
      tab = &DOPRI8;
      break;
   default:
      return NULL;
   }
   if(tab==NULL) return NULL;
   return rk_init_context_from_tab(tab, initial_condition, time_info, rhs, callback);
}

#define CLONE(dest, src) context->dest.clone(context->dest.data, context->src.data)
#define ADD_WITH_WEIGHT(dest, w, src)                                                              \
   context->dest.add_with_weight(context->dest.data, (w), context->src.data)
void rk_step(RungeKuttaContext *context)
{

   // context->_temp_step.clone(context->_temp_step.data, context->solution.data);
   CLONE(_temp_step, solution);

   for (size_t i = 0; i < context->tableau.order; i++) {
      CLONE(_ki[i], solution);
      for (size_t j = 0; j < i; j++) {
         double w = context->tableau._aij[i][j];
         if (w == 0.) continue;
         ADD_WITH_WEIGHT(_ki[i], w, _ki[j]);
      }

      context->_rhs(context->_t + context->_dt * context->tableau._ci[i], &context->_ki[i]);

      context->_ki[i].scalar_mult(context->_ki[i].data, context->_dt);

      ADD_WITH_WEIGHT(_temp_step, context->tableau._bi[i], _ki[i]);
   }

   CLONE(solution, _temp_step);
}
#undef CLONE
#undef ADD_WITH_WEIGHT

void rk_evolve(RungeKuttaContext *context)
{
   for (size_t i = 0; i < context->_time_info.size; i++) {
      context->_t  = context->_time_info._ts[i];
      context->_dt = context->_time_info._dt[i];
      for (size_t j = 0; j < context->_time_info._n_step[i]; j++) {
         rk_step(context);
         if (context->_callback != NULL && context->_cb_each_step) {
            context->_callback(context->_t, &context->_time_info, &(context->solution));
         }
         context->_t += context->_dt;
      }

      if (context->_callback != NULL) {
         context->_callback(context->_t, &context->_time_info, &(context->solution));
      }

      if (fabs(context->_t - context->_time_info._ts[i + 1]) > 1.0e-12) {
         fprintf(stderr, "RungeKutta: time steps do not match: %.6e. This is a bug.",
                 fabs(context->_t - context->_time_info._ts[i + 1]));
      }
   }
}

#endif
#endif // RUNGE_KUTTA_H
