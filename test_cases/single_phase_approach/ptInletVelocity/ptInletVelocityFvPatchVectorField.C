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

\*---------------------------------------------------------------------------*/

#include "ptInletVelocityFvPatchVectorField.H"
#include "addToRunTimeSelectionTable.H"
#include "volFields.H"
#include "surfaceFields.H"

// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

Foam::ptInletVelocityFvPatchVectorField::
ptInletVelocityFvPatchVectorField
(
    const fvPatch& p,
    const DimensionedField<vector, volMesh>& iF
)
:
    fixedValueFvPatchField<vector>(p, iF),
    direction_(Zero),
    model_name_("")
{}

Foam::ptInletVelocityFvPatchVectorField::
ptInletVelocityFvPatchVectorField
(
    const fvPatch& p,
    const DimensionedField<vector, volMesh>& iF,
    const dictionary& dict
)
:
    fixedValueFvPatchField<vector>(p, iF, dict, false),
    direction_(dict.lookup("direction")),
    model_name_(dict.get<word>("model")),
    velocity_model_(torch::jit::load(model_name_))
{
    Info << model_name_ << "\n";
    if (dict.found("value"))
    {
        fvPatchField<vector>::operator=
        (
            vectorField("value", dict, p.size())
        );
    }
    else
    {
        // Evaluate the wall velocity
        updateCoeffs();
    }
}


Foam::ptInletVelocityFvPatchVectorField::
ptInletVelocityFvPatchVectorField
(
    const ptInletVelocityFvPatchVectorField& ptf,
    const fvPatch& p,
    const DimensionedField<vector, volMesh>& iF,
    const fvPatchFieldMapper& mapper
)
:
    fixedValueFvPatchField<vector>(ptf, p, iF, mapper),
    direction_(ptf.direction_),
    model_name_(ptf.model_name_),
    velocity_model_(ptf.velocity_model_)
{}


Foam::ptInletVelocityFvPatchVectorField::
ptInletVelocityFvPatchVectorField
(
    const ptInletVelocityFvPatchVectorField& rwvpvf
)
:
    fixedValueFvPatchField<vector>(rwvpvf),
    direction_(rwvpvf.direction_),
    model_name_(rwvpvf.model_name_),
    velocity_model_(rwvpvf.velocity_model_)
{}


Foam::ptInletVelocityFvPatchVectorField::
ptInletVelocityFvPatchVectorField
(
    const ptInletVelocityFvPatchVectorField& rwvpvf,
    const DimensionedField<vector, volMesh>& iF
)
:
    fixedValueFvPatchField<vector>(rwvpvf, iF),
    direction_(rwvpvf.direction_),
    model_name_(rwvpvf.model_name_),
    velocity_model_(rwvpvf.velocity_model_)
{}


// * * * * * * * * * * * * * * * Member Functions  * * * * * * * * * * * * * //

void Foam::ptInletVelocityFvPatchVectorField::updateCoeffs()
{
    if (updated())
    {
        return;
    }

    // compute tangent vector
    const polyMesh& mesh = this->internalField().mesh();
    const Time& t = mesh.time();

    auto time = torch::zeros({1}, torch::kFloat64);
    time[0] = t.value();
    std::vector<torch::jit::IValue> feature{time};
    auto velocity = velocity_model_.forward(feature).toTensor();
    auto velocityAcc = velocity.item<double>();
    vector v_in;
    v_in = direction_ * velocityAcc;
    vectorField::operator=(v_in);
    fixedValueFvPatchVectorField::updateCoeffs();
}


void Foam::ptInletVelocityFvPatchVectorField::write(Ostream& os) const
{
    fvPatchVectorField::write(os);
    os.writeEntry("direction", direction_);
    os.writeEntry<word>("model", model_name_);
    writeEntry("value", os);
}


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{
    makePatchTypeField
    (
        fvPatchVectorField,
        ptInletVelocityFvPatchVectorField
    );
}

// ************************************************************************* //
