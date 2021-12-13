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
    Foam::ptInletVelocityFvPatchVectorField

Group
    grpWallBoundaryConditions grpGenericBoundaryConditions

Description
    This boundary condition provides an inflow boundary condition
    for the velocity field. The magnitude of the velocity is
    computed from a PyTorch model.

Usage
    \table
        Property     | Description                  | Required | Default value
        direction    | direction of inlet velocity | yes       |
        model_name   | name of the PyTorch model   | yes       |
    \endtable

    Example of the boundary condition specification:
    \verbatim
    <patchName>
    {
        type            ptInletVelocity;
        direction       (0 -1  0);
        model_name      "rise_vel_CB1.ts";
    }
    \endverbatim

See also
    Foam::fixedValueFvPatchField

SourceFiles
    ptInletVelocityFvPatchVectorField.C

\*---------------------------------------------------------------------------*/

#ifndef ptInletVelocityFvPatchVectorField_H
#define ptInletVelocityFvPatchVectorField_H

#include <torch/script.h>
#include "fixedValueFvPatchFields.H"
#include "mathematicalConstants.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

/*---------------------------------------------------------------------------*\
        Class ptInletVelocityFvPatchVectorField Declaration
\*---------------------------------------------------------------------------*/

class ptInletVelocityFvPatchVectorField
:
    public fixedValueFvPatchVectorField
{
    // Private data

        //- rise direction
        vector direction_;

        //- name of the PyTorch model
        word model_name_;

        //- PyTorch model
        torch::jit::script::Module velocity_model_;


public:

    //- Runtime type information
    TypeName("ptInletVelocity");


    // Constructors

        //- Construct from patch and internal field
        ptInletVelocityFvPatchVectorField
        (
            const fvPatch&,
            const DimensionedField<vector, volMesh>&
        );

        //- Construct from patch, internal field and dictionary
        ptInletVelocityFvPatchVectorField
        (
            const fvPatch&,
            const DimensionedField<vector, volMesh>&,
            const dictionary&
        );

        //- Construct by mapping given ptInletVelocityFvPatchVectorField
        //  onto a new patch
        ptInletVelocityFvPatchVectorField
        (
            const ptInletVelocityFvPatchVectorField&,
            const fvPatch&,
            const DimensionedField<vector, volMesh>&,
            const fvPatchFieldMapper&
        );

        //- Construct as copy
        ptInletVelocityFvPatchVectorField
        (
            const ptInletVelocityFvPatchVectorField&
        );

        //- Construct and return a clone
        virtual tmp<fvPatchVectorField> clone() const
        {
            return tmp<fvPatchVectorField>
            (
                new ptInletVelocityFvPatchVectorField(*this)
            );
        }

        //- Construct as copy setting internal field reference
        ptInletVelocityFvPatchVectorField
        (
            const ptInletVelocityFvPatchVectorField&,
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
                new ptInletVelocityFvPatchVectorField(*this, iF)
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
