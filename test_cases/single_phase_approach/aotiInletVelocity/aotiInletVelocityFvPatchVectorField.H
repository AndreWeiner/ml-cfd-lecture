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
    Foam::aotiInletVelocityFvPatchVectorField

Group
    grpWallBoundaryConditions grpGenericBoundaryConditions

Description
    This boundary condition provides an inflow boundary condition
    for the velocity field. The magnitude of the velocity is
    computed from a AOTInductor model.

Usage
    \table
        Property     | Description                  | Required | Default value
        direction    | direction of inlet velocity | yes       |
        model        | name of the AOTInductor model   | yes       |
    \endtable

    Example of the boundary condition specification:
    \verbatim
    <patchName>
    {
        type            aotiInletVelocity;
        direction       (0 -1  0);
        model           "ub.pt2";
    }
    \endverbatim

See also
    Foam::fixedValueFvPatchField

SourceFiles
    aotiInletVelocityFvPatchVectorField.C

\*---------------------------------------------------------------------------*/

#ifndef aotiInletVelocityFvPatchVectorField_H
#define aotiInletVelocityFvPatchVectorField_H

#include "AotiModel.H"
#include <memory>
#include "fixedValueFvPatchFields.H"
#include "mathematicalConstants.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

/*---------------------------------------------------------------------------*\
        Class aotiInletVelocityFvPatchVectorField Declaration
\*---------------------------------------------------------------------------*/

class aotiInletVelocityFvPatchVectorField
:
    public fixedValueFvPatchVectorField
{
    // Private data

        //- rise direction
        vector direction_;

        //- name of the AOTInductor model
        word model_name_;

        //- AOTInductor model
        std::shared_ptr<AotiModel> velocity_model_;


public:

    using fixedValueFvPatchVectorField::operator=;

    //- Runtime type information
    TypeName("aotiInletVelocity");


    // Constructors

        //- Construct from patch and internal field
        aotiInletVelocityFvPatchVectorField
        (
            const fvPatch&,
            const DimensionedField<vector, volMesh>&
        );

        //- Construct from patch, internal field and dictionary
        aotiInletVelocityFvPatchVectorField
        (
            const fvPatch&,
            const DimensionedField<vector, volMesh>&,
            const dictionary&
        );

        //- Construct by mapping given aotiInletVelocityFvPatchVectorField
        //  onto a new patch
        aotiInletVelocityFvPatchVectorField
        (
            const aotiInletVelocityFvPatchVectorField&,
            const fvPatch&,
            const DimensionedField<vector, volMesh>&,
            const fvPatchFieldMapper&
        );

        //- Construct as copy
        aotiInletVelocityFvPatchVectorField
        (
            const aotiInletVelocityFvPatchVectorField&
        );

        //- Construct and return a clone
        virtual tmp<fvPatchVectorField> clone() const
        {
            return tmp<fvPatchVectorField>
            (
                new aotiInletVelocityFvPatchVectorField(*this)
            );
        }

        //- Construct as copy setting internal field reference
        aotiInletVelocityFvPatchVectorField
        (
            const aotiInletVelocityFvPatchVectorField&,
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
                new aotiInletVelocityFvPatchVectorField(*this, iF)
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
