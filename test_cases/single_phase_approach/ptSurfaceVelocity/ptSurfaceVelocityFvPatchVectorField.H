/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     |
    \\  /    A nd           | Copyright (C) 2004-2010 OpenCFD Ltd.
     \\/     M anipulation  |
-------------------------------------------------------------------------------
                            | Copyright (C) 2011-2016 OpenFOAM Foundation
-------------------------------------------------------------------------------
License
    This file is part of OpenFOAM.

    OpenFOAM is free software: you can redistribute it and/or modify it
    under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    OpenFOAM is distributed in the hope that it will be useful, but WITHOUT
    ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
    FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
    for more details.

    You should have received a copy of the GNU General Public License
    along with OpenFOAM.  If not, see <http://www.gnu.org/licenses/>.

Class
    Foam::ptSurfaceVelocityFvPatchVectorField

Group
    grpWallBoundaryConditions grpGenericBoundaryConditions

Description
    This boundary condition provides a velocity boundary condition emulating
    the interface velocity at a bubble interface. The velocity is evaluated
    based on the x and y coordinate of each face center on the boundary patch.
    A PyTorch model returns the tangential velocity (scalar product of
    tangent and velocity vector) for a given point (x,y).

Usage
    \table
        Property     | Description             | Required    | Default value
        origin       | bubble center of mass in Cartesian coordinates | yes |
        axis         | rise direction          | yes |
        normal       | plane normal vector     | yes |
        modelName    | name of the PyTorch velocity model | yes |
    \endtable

    Example of the boundary condition specification:
    \verbatim
    <patchName>
    {
        type            ptSurfaceVelocity;
        origin          (0 0 0);
        axis            (0 1 0);
        normal          (0 0 1);
        modelName       "tv_CB1.ts";
    }
    \endverbatim

See also
    Foam::fixedValueFvPatchField

SourceFiles
    ptSurfaceVelocityFvPatchVectorField.C

\*---------------------------------------------------------------------------*/

#ifndef ptSurfaceVelocityFvPatchVectorField_H
#define ptSurfaceVelocityFvPatchVectorField_H

#include <torch/script.h>
#include "fixedValueFvPatchFields.H"
#include "mathematicalConstants.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

/*---------------------------------------------------------------------------*\
        Class ptSurfaceVelocityFvPatchVectorField Declaration
\*---------------------------------------------------------------------------*/

class ptSurfaceVelocityFvPatchVectorField
:
    public fixedValueFvPatchVectorField
{
    // Private data

        //- bubble center of mass
        vector origin_;

        //- rise direction
        vector axis_;

        //- vector normal to 2D mesh / wedge
        vector normal_;

        //- name of the PyTorch velocity model
        word model_name_;

        //- PyTorch model for the surface velocity
        torch::jit::script::Module velocity_model_;

public:

    //- Runtime type information
    TypeName("ptSurfaceVelocity");


    // Constructors

        //- Construct from patch and internal field
        ptSurfaceVelocityFvPatchVectorField
        (
            const fvPatch&,
            const DimensionedField<vector, volMesh>&
        );

        //- Construct from patch, internal field and dictionary
        ptSurfaceVelocityFvPatchVectorField
        (
            const fvPatch&,
            const DimensionedField<vector, volMesh>&,
            const dictionary&
        );

        //- Construct by mapping given ptSurfaceVelocityFvPatchVectorField
        //  onto a new patch
        ptSurfaceVelocityFvPatchVectorField
        (
            const ptSurfaceVelocityFvPatchVectorField&,
            const fvPatch&,
            const DimensionedField<vector, volMesh>&,
            const fvPatchFieldMapper&
        );

        //- Construct as copy
        ptSurfaceVelocityFvPatchVectorField
        (
            const ptSurfaceVelocityFvPatchVectorField&
        );

        //- Construct and return a clone
        virtual tmp<fvPatchVectorField> clone() const
        {
            return tmp<fvPatchVectorField>
            (
                new ptSurfaceVelocityFvPatchVectorField(*this)
            );
        }

        //- Construct as copy setting internal field reference
        ptSurfaceVelocityFvPatchVectorField
        (
            const ptSurfaceVelocityFvPatchVectorField&,
            const DimensionedField<vector, volMesh>&
        );

        //- Construct and return a clone setting internal field reference
        virtual tmp<fvPatchVectorField> clone
        (
             const DimensionedField<vector, volMesh>& iF
        ) const
        {
            return tmp<fvPatchVectorField>
            (
                new ptSurfaceVelocityFvPatchVectorField(*this, iF)
            );
        }



    // Member functions

        //- Update the coefficients associated with the patch field
        virtual void updateCoeffs();

        //- Write
        virtual void write(Ostream&) const;
};


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif

// ************************************************************************* //
