/**
This solver is a combination of
 - bubble.c: http://basilisk.fr/src/examples/bubble.c
 - rising.c: http://basilisk.fr/src/test/rising.c

The functions to write the vtk output comes from acstilo's sandbox:
http://basilisk.fr/sandbox/acastillo/readme
*/
#include "axi.h"
#include "navier-stokes/centered.h"
#include "navier-stokes/perfs.h"
#define mu(f)  (1./(clamp(f,0,1)*(1./mu1 - 1./mu2) + 1./mu2))
#include "two-phase.h"
#include "tension.h"
#include "maxruntime.h"
#include "fractions.h"
#include "output_vtu_foreach.h"
#include "fractions_output.h"

/**
Default values
*/
#define RHOR 1000
#define MUR 100
#define WIDTH 120.0
#define X0 3.5
int LEVEL = 13;
int LEVEL_INIT = 13;
double MAXTIME = 10;
double Bo = 1.0;
double Ga = 100.0;
int counter = 0;

/**
Boundary conditions
*/

u.t[right] = dirichlet(0);
u.t[left]  = dirichlet(0);
uf.n[bottom] = 0.;
uf.n[top] = 0.;

/**
The main function takes four parameters:
 - the maximum refinement level
 - the end time
 - the Bond number
 - the Galilei number
*/

int main (int argc, char * argv[]) {
  // crete folders to write results
  system("mkdir -p vtu");
  system("mkdir -p plic");
  system("mkdir -p dump");
  maxruntime (&argc, argv);
  if (argc > 1)
    LEVEL = atoi (argv[1]);
  if (argc > 2)
    MAXTIME = atoi (argv[2]);
  if (argc > 3)
    Bo = atof(argv[3]);
  if (argc > 4)
    Ga = atof(argv[4]);
  size (WIDTH);
  origin (0, 0);
  init_grid (128);
  rho1 = 1.0;
  rho2 = 1.0/RHOR;
  mu1 = 1.0/Ga;
  mu2 = 1.0/(MUR*Ga);
  f.sigma = 1.0/Bo;
  TOLERANCE = 1e-4;
  run();
}

event init (t = 0) {
  if (!restore (file = "restart")) {
    refine (sq(x-X0) + sq(y) - sq(0.75) < 0 && level < LEVEL_INIT);
    fraction (f, sq(x-X0) + sq(y) - sq(0.5));
  }
}

event acceleration (i++) {
  face vector av = a;
  foreach_face(x)
    av.x[] -= 1.;
}

event adapt (i++) {
  double uemax = 1e-2;
  adapt_wavelet ({f,u}, (double[]){1.0e-3,uemax,uemax}, LEVEL, 5);
}

event logfile (i += 10) {
  double xb = 0., yb = 0., zb = 0., sb = 0.;
  double vbx = 0., vby = 0., vbz = 0.;
  double area = 0.;
  foreach(reduction(+:xb) reduction(+:yb) reduction(+:zb)
	  reduction(+:vbx) reduction(+:vby) reduction(+:vbz)
	  reduction(+:sb)) {
    double dv = (1. - f[])*dv();
    xb += x*dv;
    yb += y*dv;
    zb += z*dv;
    vbx += u.x[]*dv;
    vby += u.y[]*dv;
    vbz += u.z[]*dv;
    sb += dv;
  }
  area = interface_area (f);
  fprintf (ferr,
	   "%.8f %.8f %.8f %.8f %.8f %.8f %.8f %.8f %.8f \n",
	   t, sb,
	   xb/sb, yb/sb, zb/sb,
	   vbx/sb, vby/sb, vbz/sb,
     area );
  fflush (ferr);


}

event dumpstatus (i += 1000)
{
  char name[80];
  sprintf (name, "dump/dump-%06d", i);
  dump (file = name);
#if _MPI
  MPI_Barrier (MPI_COMM_WORLD);
#endif
  if (pid() == 0) {
    char copy_command[100];
    sprintf (copy_command, "ln -sf dump/dump-%06d restart", i);
    system(copy_command);
  }
}

event snapshot (t = 0; t += MAXTIME/100.0; t <= MAXTIME)
{
  scalar l_ref[];
  foreach()
    l_ref[] = level;

  char subname[80];
  FILE * fp ;
  sprintf(subname, "vtu/bubble_%6.6d_n%3.3d.vtu", counter, pid());
  fp = fopen(subname, "w");
  output_vtu_bin_foreach ((scalar *) {f, l_ref}, (vector *) {u}, t, fp, false);
  fclose (fp);

  char name[80];
  sprintf(name, "vtu/bubble_%6.6d.pvtu", counter);
  char base_name[80];
  sprintf(base_name, "bubble_%6.6d", counter);
  fp = fopen(name, "w");
  output_pvtu_bin ((scalar *) {f, l_ref}, (vector *) {u}, t, fp, base_name);
  fclose (fp);

  char plic_subname[80];
  sprintf(plic_subname, "vtu/plic_%6.6d_n%3.3d.vtu", counter, pid());
  fp = fopen(plic_subname, "w");
  output_vtu_w_fielddata (f, fp, (scalar *) {f}, (vector *) {u});
  fclose (fp);

  char plic_pvtu_name[80];
  sprintf(plic_pvtu_name, "vtu/plic_%6.6d.pvtu", counter);
  char plic_base_name[80];
  sprintf(plic_base_name, "plic_%6.6d", counter);
  fp = fopen(plic_pvtu_name, "w");
  output_plic_pvtu_bin ((scalar *) {f}, (vector *) {u}, t, fp, plic_base_name);
  fclose (fp);

  char interface[80];
  sprintf(interface, "plic/points_%6.6d_n%3.3d.txt", counter, pid());
  fp = fopen(interface, "w");
  output_interface_data(f, u, fp);
  //output_facets(f, fp);
  fclose (fp);
  counter++;
}
