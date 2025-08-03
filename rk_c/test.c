#define RK_IMPLEMENTATION
#include "runge_kutta.h"

typedef struct {
      double x0, v0;
} HO;

void init_HO(void **dest)
{
   *dest   = (HO *)calloc(1, sizeof(HO));
   HO *tmp = (HO *)(*dest);
   tmp->x0 = 1.0;
   tmp->v0 = 0.0;
}

void free_HO(void **dest)
{
   free(*dest);
   dest = NULL;
}

void clone_HO(void *dest, void *src)
{
   HO *dest_ho = (HO *)dest;
   HO *src_ho  = (HO *)src;
   dest_ho->x0 = src_ho->x0;
   dest_ho->v0 = src_ho->v0;
}

void add_with_weight_HO(void *dest, double w, void *src)
{
   HO *dest_ho = (HO *)dest;
   HO *src_ho  = (HO *)src;
   dest_ho->x0 += w * src_ho->x0;
   dest_ho->v0 += w * src_ho->v0;
}

void scalar_mult_HO(void *dest, double s)
{
   HO *dest_ho = (HO *)dest;
   dest_ho->x0 *= s;
   dest_ho->v0 *= s;
}

void rhs_HO(double t, Solution *sol)
{
   (void)t;
   HO *ho   = (HO *)sol->data;
   double v = ho->v0;
   ho->v0   = -ho->x0;
   ho->x0   = v;
}

void callback_HO(double t, const TimeInfo *const tinfo, Solution * sol) 
{
   (void)tinfo;
   printf("%.4f\t%.16e\n", t, ((HO*)(sol->data))->x0);
}

int main(void)
{
   // rk_print_tableau(&DOPRI8);
   Solution ini = {
       .clone           = clone_HO,
       .add_with_weight = add_with_weight_HO,
       .scalar_mult     = scalar_mult_HO,
       .init_data       = init_HO,
       .free_data       = free_HO,
   };
   init_HO(&ini.data);

   TimeInfo t_info = rk_generate_time_info((double[]){0., 10.}, 2, 0.1, 10);

   RungeKuttaContext *context =
       rk_init_context(DOPRI8_T, &ini, t_info, rhs_HO, callback_HO);


   rk_evolve(context);

   rk_free_runge_kutta_context(&context);
   return 0;
}
