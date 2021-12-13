#!/usr/bin/bash

DIR=$(pwd)

# applications
## calculation of local and global Sherwood numbers
cd calcSh/ && wclean && wmake && cd $DIR
## incompressible solver with additional solution of scalar transport equations
cd scalarPimpleFoam/ && wclean && wmake && cd $DIR

# boundary conditions
## inlet velocity
cd ptInletVelocity/ && wclean && wmake && cd $DIR
## bubble surface velocity
cd ptSurfaceVelocity/ && wclean && wmake && cd $DIR
## bubble surface displacement
cd ptBoundaryDisplacement/ && wclean && wmake && cd $DIR