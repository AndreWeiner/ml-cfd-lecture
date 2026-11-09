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
    Foam::aotiDisplacementPointPatchVectorField

Description
    This boundary condition provides the time dependent position vector of
    point on the surface of a rising bubble.

Usage
    \table
        Property     | Description                  | Required | Default value
        center       | bubble center of mass        | yes      |
        direction    | rise direction               | yes      |
        normal       | plane normal vector          | yes      |
        radius       | initial radius               | yes      |
        model    | name of the AOTInductor model    | yes      |
    \endtable

    Example of the boundary condition specification:
    \verbatim
    <patchName>
    {
        type            aotiBoundaryDisplacement;
        center          (0 0 0);
        direction       (0 1 0);
        normal          (0 0 1);
        radius          0.5;
        model       "r.pt2";
    }
    \endverbatim


SourceFiles
    aotiDisplacementPointPatchVectorField.C

\*---------------------------------------------------------------------------*/

#ifndef aotiDisplacementPointPatchVectorField_H
#define aotiDisplacementPointPatchVectorField_H

#include "AotiModel.H"
#include <memory>
#include "fixedValuePointPatchField.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

/*---------------------------------------------------------------------------*\
        Class aotiDisplacementPointPatchVectorField Declaration
\*---------------------------------------------------------------------------*/

class aotiDisplacementPointPatchVectorField
:
    public fixedValuePointPatchField<vector>
{
    // Private data

        vector center_;
        vector direction_;
        vector normal_;
        scalar radius_;
        word model_name_;
        std::shared_ptr<AotiModel> shape_model_;


public:

    using fixedValuePointPatchField<vector>::operator=;

    //- Runtime type information
    TypeName("aotiBoundaryDisplacement");


    // Constructors

        //- Construct from patch and internal field
        aotiDisplacementPointPatchVectorField
        (
            const pointPatch&,
            const DimensionedField<vector, pointMesh>&
        );

        //- Construct from patch, internal field and dictionary
        aotiDisplacementPointPatchVectorField
        (
            const pointPatch&,
            const DimensionedField<vector, pointMesh>&,
            const dictionary&
        );

        //- Construct by mapping given patchField<vector> onto a new patch
        aotiDisplacementPointPatchVectorField
        (
            const aotiDisplacementPointPatchVectorField&,
            const pointPatch&,
            const DimensionedField<vector, pointMesh>&,
            const pointPatchFieldMapper&
        );

        //- Construct and return a clone
        virtual autoPtr<pointPatchField<vector>> clone() const
        {
            return autoPtr<pointPatchField<vector>>
            (
                new aotiDisplacementPointPatchVectorField
                (
                    *this, this->internalField()
                )
            );
        }

        //- Construct as copy setting internal field reference
        aotiDisplacementPointPatchVectorField
        (
            const aotiDisplacementPointPatchVectorField&,
            const DimensionedField<vector, pointMesh>&
        );

        //- Construct and return a clone setting internal field reference
        virtual autoPtr<pointPatchField<vector>> clone
        (
            const DimensionedField<vector, pointMesh>& iF
        ) const
        {
            return autoPtr<pointPatchField<vector>>
            (
                new aotiDisplacementPointPatchVectorField
                (
                    *this,
                    iF
                )
            );
        }


    // Member functions

        // Evaluation functions

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
