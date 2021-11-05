/**
# Additional output functions to generate surface values */

#include "geometry.h"
#include "fractions.h"

#if dimension == 1
coord mycs (Point point, scalar c) {
  coord n = {1.};
  return n;
}
#elif dimension == 2
# include "myc2d.h"
#else // dimension == 3
# include "myc.h"
#endif



void output_ply (struct OutputFacets p)
{

  #if defined(_OPENMP)
  		int num_omp = omp_get_max_threads();
  		omp_set_num_threads(1);
  #endif
  scalar c = p.c;
  face vector s = p.s;
  if (!p.fp) p.fp = stdout;

  // print header text
  fputs ("ply\n", p.fp);
  fputs ("format ascii 1.0\n", p.fp);

  int nverts = 0;
  int nfacets = 0;

  foreach()
    if (c[] > 1e-6 && c[] < 1. - 1e-6) {
      coord n;
      if (!s.x.i)
	// compute normal from volume fraction
	n = mycs (point, c);
      else {
	// compute normal from face fractions
	double nn = 0.;
	foreach_dimension() {
	  n.x = s.x[] - s.x[1];
	  nn += fabs(n.x);
	}
	assert (nn > 0.);
	foreach_dimension()
	  n.x /= nn;
      }
      double alpha = plane_alpha (c[], n);

      coord v[12];
#if dimension == 2
      int m = facets (n, alpha, v);
#else // dimension == 3
      int m = facets (n, alpha, v, 1.);
#endif
      //int m = facets (n, alpha, v, 1.);
      for (int i = 0; i < m; i++) {
        nverts ++;
        }
      if (m > 0) {
	nfacets ++;
        }
    }

  fprintf (p.fp, "element vertex %i\n", nverts);
  fputs ("property float x\n", p.fp);
  fputs ("property float y\n", p.fp);
  fputs ("property float z\n", p.fp);
  fprintf (p.fp, "element face %i\n", nfacets);
  fputs ("property list uchar int vertex_index\n", p.fp);
  fputs ("end_header\n", p.fp);

  int facet_num[nfacets];

  int ifacet = 0;
  int ivert = 0;

  foreach()
    if (c[] > 1e-6 && c[] < 1. - 1e-6) {
      coord n;
      if (!s.x.i)
	// compute normal from volume fraction
	n = mycs (point, c);
      else {
	// compute normal from face fractions
	double nn = 0.;
	foreach_dimension() {
	  n.x = s.x[] - s.x[1];
	  nn += fabs(n.x);
	}
	assert (nn > 0.);
	foreach_dimension()
	  n.x /= nn;
      }
      double alpha = plane_alpha (c[], n);

      coord v[12];
#if dimension == 2
      int m = facets (n, alpha, v);
#else // dimension == 3
      int m = facets (n, alpha, v, 1.);
#endif
      //int m = facets (n, alpha, v, 1.);
      for (int i = 0; i < m; i++) {
	fprintf (p.fp, "%g %g %g\n",
		 x + v[i].x*Delta, y + v[i].y*Delta, z + v[i].z*Delta);
        }
      if (m > 0) {
	facet_num[ifacet] = m;
	ifacet ++;
        }
    }

  // print face list
  for (ifacet = 0; ifacet < nfacets; ifacet++) {
    fprintf (p.fp, "%i ", facet_num[ifacet]);
    for (int iv = 0; iv < facet_num[ifacet]; iv ++) {
      fprintf (p.fp, "%i ", ivert);
      ivert ++;
      }
    fputc ('\n', p.fp);
    }

  fflush (p.fp);
  #if defined(_OPENMP)
  	omp_set_num_threads(num_omp);
  #endif
}


void output_vtu (struct OutputFacets p)
{
  #if defined(_OPENMP)
  		int num_omp = omp_get_max_threads();
  		omp_set_num_threads(1);
  #endif
  scalar c = p.c;
  face vector s = p.s;
  if (!p.fp) p.fp = stdout;

  // print header text
  fputs ("<?xml version=\"1.0\"?>\n", p.fp);
  fputs ("<VTKFile type=\"UnstructuredGrid\" version=\"0.1\" byte_order=\"LittleEndian\">\n", p.fp);
  fputs ("  <UnstructuredGrid>\n", p.fp);

  int nverts = 0;
  int nfacets = 0;

  foreach()
    if (c[] > 1e-6 && c[] < 1. - 1e-6) {
      coord n;
      if (!s.x.i)
	// compute normal from volume fraction
	n = mycs (point, c);
      else {
	// compute normal from face fractions
	double nn = 0.;
	foreach_dimension() {
	  n.x = s.x[] - s.x[1];
	  nn += fabs(n.x);
	}
	assert (nn > 0.);
	foreach_dimension()
	  n.x /= nn;
      }
      double alpha = plane_alpha (c[], n);

      coord v[12];
#if dimension == 2
      int m = facets (n, alpha, v);
#else // dimension == 3
      int m = facets (n, alpha, v, 1.);
#endif
      //int m = facets (n, alpha, v, 1.);
      for (int i = 0; i < m; i++) {
        nverts ++;
        }
      if (m > 0) {
	nfacets ++;
        }
    }

  fprintf (p.fp, "    <Piece NumberOfPoints=\"%i\" NumberOfCells=\"%i\">\n", nverts, nfacets);
  fputs ("      <Points>\n", p.fp);
  fputs ("        <DataArray type=\"Float32\" Name=\"vertices\" NumberOfComponents=\"3\" format=\"ascii\">\n", p.fp);

  int offsets[nfacets];

  int ifacet = 0;
  int offset = 0;

  foreach()
    if (c[] > 1e-6 && c[] < 1. - 1e-6) {
      coord n;
      if (!s.x.i)
	// compute normal from volume fraction
	n = mycs (point, c);
      else {
	// compute normal from face fractions
	double nn = 0.;
	foreach_dimension() {
	  n.x = s.x[] - s.x[1];
	  nn += fabs(n.x);
	}
	assert (nn > 0.);
	foreach_dimension()
	  n.x /= nn;
      }
      double alpha = plane_alpha (c[], n);

      coord v[12];
#if dimension == 2
      int m = facets (n, alpha, v);
#else // dimension == 3
      int m = facets (n, alpha, v, 1.);
#endif
      //int m = facets (n, alpha, v, 1.);
      for (int i = 0; i < m; i++) {
	fprintf (p.fp, "%g %g %g ",
		 x + v[i].x*Delta, y + v[i].y*Delta, z + v[i].z*Delta);
        }
      if (m > 0) {
        offset += m;
        offsets[ifacet] = offset;
	ifacet ++;
        }
    }


  fputs ("        </DataArray>\n", p.fp);
  fputs ("      </Points>\n", p.fp);
  fputs ("      <Cells>\n", p.fp);

  fputs ("        <DataArray type=\"Int32\" Name=\"connectivity\" format=\"ascii\">\n", p.fp);

  // print vert numbers
  for (int ivert = 0; ivert < nverts; ivert++)
    fprintf (p.fp, "%i ", ivert);

  fputs ("        </DataArray>\n", p.fp);
  fputs ("        <DataArray type=\"Int32\" Name=\"offsets\" format=\"ascii\">\n", p.fp);

  // print offsets
  for (ifacet = 0; ifacet < nfacets; ifacet++)
    fprintf (p.fp, "%i ", offsets[ifacet]);

  fputs ("        </DataArray>\n", p.fp);
  fputs ("        <DataArray type=\"UInt8\" Name=\"types\" format=\"ascii\">\n", p.fp);

  // print cell type list
  for (ifacet = 0; ifacet < nfacets; ifacet++)
    fprintf (p.fp, "7 ");

  fputs ("        </DataArray>\n", p.fp);
  fputs ("      </Cells>\n", p.fp);
  fputs ("      <PointData>\n", p.fp);
  fputs ("      </PointData>\n", p.fp);
  fputs ("      <CellData>\n", p.fp);
  fputs ("      </CellData>\n", p.fp);
  fputs ("    </Piece>\n", p.fp);
  fputs ("  </UnstructuredGrid>\n", p.fp);
  fputs ("</VTKFile>\n", p.fp);

  fflush (p.fp);
  #if defined(_OPENMP)
  	omp_set_num_threads(num_omp);
  #endif
}


struct OutputFacets_scalar {
  scalar c;
  FILE * fp;     // optional: default is stdout
  scalar * list;  // List of scalar fields to include when writing vtu surface to file
  vector * vlist; // List of vector fields to include.
  face vector s; // optional: default is none
};

/*
This function writes one XML file which allows to read the *.vtu files generated
by output_vtu_w_fielddata() when used in MPI. Tested in (quad- and oct-)trees
using MPI.
*/
void output_plic_pvtu_bin (scalar * list, vector * vlist, int n, FILE * fp, char * subname)
{
    fputs ("<?xml version=\"1.0\"?>\n"
    "<VTKFile type=\"PUnstructuredGrid\" version=\"1.0\" byte_order=\"LittleEndian\" header_type=\"UInt64\">\n", fp);
    fputs ("\t <PUnstructuredGrid GhostLevel=\"0\">\n", fp);
    fputs ("\t\t\t <PCellData Scalars=\"scalars\">\n", fp);
    for (scalar s in list) {
      fprintf (fp,"\t\t\t\t <PDataArray type=\"Float64\" Name=\"%s\" format=\"appended\">\n", s.name);
      fputs ("\t\t\t\t </PDataArray>\n", fp);
    }
    for (vector v in vlist) {
      fprintf (fp,"\t\t\t\t <PDataArray type=\"Float64\" NumberOfComponents=\"3\" Name=\"%s\" format=\"appended\">\n", v.x.name);
      fputs ("\t\t\t\t </PDataArray>\n", fp);
    }
    fputs ("\t\t\t </PCellData>\n", fp);
    fputs ("\t\t\t <PPoints>\n", fp);
    fputs ("\t\t\t\t <PDataArray type=\"Float32\" NumberOfComponents=\"3\" format=\"ascii\">\n", fp);
    fputs ("\t\t\t\t </PDataArray>\n", fp);
    fputs ("\t\t\t </PPoints>\n", fp);

    for (int i = 0; i < npe(); i++)
      fprintf (fp, "<Piece Source=\"%s_n%3.3d.vtu\"/> \n", subname, i);

    fputs ("\t </PUnstructuredGrid>\n", fp);
    fputs ("</VTKFile>\n", fp);
}

void output_vtu_w_fielddata (struct OutputFacets_scalar p)
{
  #if defined(_OPENMP)
  		int num_omp = omp_get_max_threads();
  		omp_set_num_threads(1);
  #endif
  scalar c = p.c;
  face vector s = p.s;
  if (!p.fp) p.fp = stdout;

  // print header text
  fputs ("<?xml version=\"1.0\"?>\n", p.fp);
  fputs ("<VTKFile type=\"UnstructuredGrid\" version=\"0.1\" byte_order=\"LittleEndian\">\n", p.fp);
  fputs ("  <UnstructuredGrid>\n", p.fp);

  int nverts = 0;
  int nfacets = 0;

  foreach()
    if (c[] > 1e-6 && c[] < 1. - 1e-6) {
      coord n;
      if (!s.x.i)
	// compute normal from volume fraction
	n = mycs (point, c);
      else {
	// compute normal from face fractions
	double nn = 0.;
	foreach_dimension() {
	  n.x = s.x[] - s.x[1];
	  nn += fabs(n.x);
	}
	assert (nn > 0.);
	foreach_dimension()
	  n.x /= nn;
      }
      double alpha = plane_alpha (c[], n);

      coord v[12];
#if dimension == 2
      int m = facets (n, alpha, v);
#else // dimension == 3
      int m = facets (n, alpha, v, 1.);
#endif
      //int m = facets (n, alpha, v, 1.);
      for (int i = 0; i < m; i++) {
        nverts ++;
        }
      if (m > 0) {
	nfacets ++;
        }
    }
  fprintf (p.fp, "    <Piece NumberOfPoints=\"%i\" NumberOfCells=\"%i\">\n", nverts, nfacets);
//  fprintf (p.fp, "\t\t\"%i\" NumberOfCells=\"%i\">\n", nverts, nfacets);

  // Write list of scalar field values to file
  fputs ("\t\t\t <CellData Scalars=\"scalars\">\n", p.fp);
  for (scalar s in p.list) {
    fprintf (p.fp,"\t\t\t\t <DataArray type=\"Float64\" Name=\"%s\" format=\"ascii\">\n", s.name);
    foreach(){
      if (c[] > 1e-6 && c[] < 1. - 1e-6) {
        fprintf (p.fp, "%g\n", val(s));
      }
    }
    fputs ("\t\t\t\t </DataArray>\n", p.fp);
  }
  // Write list of vector field values to file
  for (vector v in p.vlist) {
    fprintf (p.fp,"\t\t\t\t <DataArray type=\"Float64\" NumberOfComponents=\"3\" Name=\"%s\" format=\"ascii\">\n", v.x.name);
    foreach(){
      if (c[] > 1e-6 && c[] < 1. - 1e-6) {
        #if dimension == 2
          fprintf (p.fp, "%g %g 0.\n", val(v.x), val(v.y));
        #endif
        #if dimension == 3
          fprintf (p.fp, "%g %g %g\n", val(v.x), val(v.y), val(v.z));
        #endif
      }
    }
    fputs ("\t\t\t\t </DataArray>\n", p.fp);
  }
  fputs ("\t\t\t </CellData>\n", p.fp);


  // Write points to file
  fputs ("      <Points>\n", p.fp);
  fputs ("        <DataArray type=\"Float32\" Name=\"vertices\" NumberOfComponents=\"3\" format=\"ascii\">\n", p.fp);

  int offsets[nfacets];

  int ifacet = 0;
  int offset = 0;

  foreach()
    if (c[] > 1e-6 && c[] < 1. - 1e-6) {
      coord n;
      if (!s.x.i)
	// compute normal from volume fraction
	n = mycs (point, c);
      else {
	// compute normal from face fractions
	double nn = 0.;
	foreach_dimension() {
	  n.x = s.x[] - s.x[1];
	  nn += fabs(n.x);
	}
	assert (nn > 0.);
	foreach_dimension()
	  n.x /= nn;
      }
      double alpha = plane_alpha (c[], n);

      coord v[12];
#if dimension == 2
      int m = facets (n, alpha, v);
#else // dimension == 3
      int m = facets (n, alpha, v, 1.);
#endif
      //int m = facets (n, alpha, v, 1.);
      for (int i = 0; i < m; i++) {
	fprintf (p.fp, "%g %g %g ",
		 x + v[i].x*Delta, y + v[i].y*Delta, z + v[i].z*Delta);
        }
      if (m > 0) {
        offset += m;
        offsets[ifacet] = offset;
	ifacet ++;
        }
    }


  fputs ("        </DataArray>\n", p.fp);
  fputs ("      </Points>\n", p.fp);
  fputs ("      <Cells>\n", p.fp);

  fputs ("        <DataArray type=\"Int32\" Name=\"connectivity\" format=\"ascii\">\n", p.fp);

  // print vert numbers
  for (int ivert = 0; ivert < nverts; ivert++)
    fprintf (p.fp, "%i ", ivert);

  fputs ("        </DataArray>\n", p.fp);
  fputs ("        <DataArray type=\"Int32\" Name=\"offsets\" format=\"ascii\">\n", p.fp);

  // print offsets
  for (ifacet = 0; ifacet < nfacets; ifacet++)
    fprintf (p.fp, "%i ", offsets[ifacet]);

  fputs ("        </DataArray>\n", p.fp);
  fputs ("        <DataArray type=\"UInt8\" Name=\"types\" format=\"ascii\">\n", p.fp);

  // print cell type list
  for (ifacet = 0; ifacet < nfacets; ifacet++)
    fprintf (p.fp, "7 ");

  fputs ("\n        </DataArray>\n", p.fp);
  fputs ("      </Cells>\n", p.fp);
  fputs ("      <PointData>\n", p.fp);
  fputs ("      </PointData>\n", p.fp);
//  fputs ("      <CellData>\n", p.fp);
//  fputs ("      </CellData>\n", p.fp);
  fputs ("    </Piece>\n", p.fp);
  fputs ("  </UnstructuredGrid>\n", p.fp);
  fputs ("</VTKFile>\n", p.fp);

  fflush (p.fp);
  #if defined(_OPENMP)
  	omp_set_num_threads(num_omp);
  #endif
}
