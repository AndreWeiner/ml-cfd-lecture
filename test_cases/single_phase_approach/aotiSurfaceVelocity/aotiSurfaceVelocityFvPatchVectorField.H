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
    Foam::aotiSurfaceVelocityFvPatchVectorField

Group
    grpWallBoundaryConditions grpGenericBoundaryConditions

Description
    This boundary condition provides a velocity boundary condition emulating
    the interface velocity at a bubble interface. The velocity is evaluated
    based on the x and y coordinate of each face center on the boundary patch.
    A AOTInductor model returns the tangential velocity (scalar product of
    tangent and velocity vector) for a given point (x,y).

Usage
    \table
        Property     | Description             | Required    | Default value
        origin       | bubble center of mass in Cartesian coordinates | yes |
        axis         | rise direction          | yes |
        normal       | plane normal vector     | yes |
        model    | name of the AOTInductor velocity model | yes |
    \endtable

    Example of the boundary condition specification:
    \verbatim
    <patchName>
    {
        type            aotiSurfaceVelocity;
        origin          (0 0 0);
        axis            (0 1 0);
        normal          (0 0 1);
        model       "ut.pt2";
    }
    \endverbatim

See also
    Foam::fixedValueFvPatchField

SourceFiles
    aotiSurfaceVelocityFvPatchVectorField.C

\*---------------------------------------------------------------------------*/

#ifndef aotiSurfaceVelocityFvPatchVectorField_H
#define aotiSurfaceVelocityFvPatchVectorField_H

#include "AotiModel.H"
#include <memory>
#include "fixedValueFvPatchFields.H"
#include "mathematicalConstants.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

/*---------------------------------------------------------------------------*\
        Class aotiSurfaceVelocityFvPatchVectorField Declaration
\*---------------------------------------------------------------------------*/

class aotiSurfaceVelocityFvPatchVectorField
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

        //- name of the AOTInductor velocity model
        word model_name_;

        //- AOTInductor model for the surface velocity
        std::shared_ptr<AotiModel> velocity_model_;

public:

    using fixedValueFvPatchVectorField::operator=;

    //- Runtime type information
    TypeName("aotiSurfaceVelocity");


    // Constructors

        //- Construct from patch and internal field
        aotiSurfaceVelocityFvPatchVectorField
        (
            const fvPatch&,
            const DimensionedField<vector, volMesh>&
        );

        //- Construct from patch, internal field and dictionary
        aotiSurfaceVelocityFvPatchVectorField
        (
            const fvPatch&,
            const DimensionedField<vector, volMesh>&,
            const dictionary&
        );

        //- Construct by mapping given aotiSurfaceVelocityFvPatchVectorField
        //  onto a new patch
        aotiSurfaceVelocityFvPatchVectorField
        (
            const aotiSurfaceVelocityFvPatchVectorField&,
            const fvPatch&,
            const DimensionedField<vector, volMesh>&,
            const fvPatchFieldMapper&
        );

        //- Construct as copy
        aotiSurfaceVelocityFvPatchVectorField
        (
            const aotiSurfaceVelocityFvPatchVectorField&
        );

        //- Construct and return a clone
        virtual tmp<fvPatchVectorField> clone() const
        {
            return tmp<fvPatchVectorField>
            (
                new aotiSurfaceVelocityFvPatchVectorField(*this)
            );
        }

        //- Construct as copy setting internal field reference
        aotiSurfaceVelocityFvPatchVectorField
        (
            const aotiSurfaceVelocityFvPatchVectorField&,
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
                new aotiSurfaceVelocityFvPatchVectorField(*this, iF)
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
