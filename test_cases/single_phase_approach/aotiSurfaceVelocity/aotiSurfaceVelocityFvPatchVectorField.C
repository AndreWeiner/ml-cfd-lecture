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

#include "aotiSurfaceVelocityFvPatchVectorField.H"
#include "addToRunTimeSelectionTable.H"
#include "volFields.H"
#include "surfaceFields.H"
#include "Time.H"
#include "polyMesh.H"
#include "fvcMeshPhi.H"

// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

Foam::aotiSurfaceVelocityFvPatchVectorField::
aotiSurfaceVelocityFvPatchVectorField
(
    const fvPatch& p,
    const DimensionedField<vector, volMesh>& iF
)
:
    fixedValueFvPatchField<vector>(p, iF),
    origin_(Zero),
    axis_(Zero),
    normal_(Zero),
    model_name_("")
{
}


Foam::aotiSurfaceVelocityFvPatchVectorField::
aotiSurfaceVelocityFvPatchVectorField
(
    const fvPatch& p,
    const DimensionedField<vector, volMesh>& iF,
    const dictionary& dict
)
:
    fixedValueFvPatchField<vector>(p, iF, dict, false),
    origin_(dict.lookup("origin")),
    axis_(dict.lookup("axis")),
    normal_(dict.lookup("normal")),
    model_name_(dict.get<word>("model")),
    velocity_model_(std::make_shared<AotiModel>(model_name_.c_str()))
{
    if (dict.found("value"))
    {
        fvPatchField<vector>::operator=(vectorField("value", dict, p.size()));
    }
    else
    {
        updateCoeffs();
    }
}


Foam::aotiSurfaceVelocityFvPatchVectorField::
aotiSurfaceVelocityFvPatchVectorField
(
    const aotiSurfaceVelocityFvPatchVectorField& ptf,
    const fvPatch& p,
    const DimensionedField<vector, volMesh>& iF,
    const fvPatchFieldMapper& mapper
)
:
    fixedValueFvPatchField<vector>(ptf, p, iF, mapper),
    origin_(ptf.origin_),
    axis_(ptf.axis_),
    normal_(ptf.normal_),
    model_name_(ptf.model_name_),
    velocity_model_(ptf.velocity_model_)
{}


Foam::aotiSurfaceVelocityFvPatchVectorField::
aotiSurfaceVelocityFvPatchVectorField
(
    const aotiSurfaceVelocityFvPatchVectorField& rwvpvf
)
:
    fixedValueFvPatchField<vector>(rwvpvf, rwvpvf.internalField()),
    origin_(rwvpvf.origin_),
    axis_(rwvpvf.axis_),
    normal_(rwvpvf.normal_),
    model_name_(rwvpvf.model_name_),
    velocity_model_(rwvpvf.velocity_model_)
{}


Foam::aotiSurfaceVelocityFvPatchVectorField::
aotiSurfaceVelocityFvPatchVectorField
(
    const aotiSurfaceVelocityFvPatchVectorField& rwvpvf,
    const DimensionedField<vector, volMesh>& iF
)
:
    fixedValueFvPatchField<vector>(rwvpvf, iF),
    origin_(rwvpvf.origin_),
    axis_(rwvpvf.axis_),
    normal_(rwvpvf.normal_),
    model_name_(rwvpvf.model_name_),
    velocity_model_(rwvpvf.velocity_model_)
{}


// * * * * * * * * * * * * * * * Member Functions  * * * * * * * * * * * * * //

void Foam::aotiSurfaceVelocityFvPatchVectorField::updateCoeffs()
{
    if (updated())
    {
        return;
    }

    const vectorField n(patch().nf()); // liquid-domain normal points into gas
    vectorField tangent(n ^ (-normal_));
    tangent /= mag(tangent) + VSMALL;
    const vectorField position(patch().Cf() - origin_);
    std::vector<double> features(2*position.size());
    forAll(position, i)
    {
        const vector planar = position[i] - normal_*(position[i] & normal_);
        const scalar axial = planar & axis_;
        const scalar radial = mag(planar - axial*axis_);
        features[2*i] = db().time().value();
        features[2*i+1] = atan2(radial, axial);
    }
    const auto ut = velocity_model_->predict(features, position.size(), 2);
    const fvMesh& mesh = internalField().mesh();
    scalarField meshFlux(position.size(), Zero);
    if (mesh.moving())
    {
        const volVectorField& U =
            static_cast<const volVectorField&>(internalField());
        // As in v2606 movingWallVelocity, use the discrete swept-volume flux.
        meshFlux = patch().patchField<surfaceScalarField>(fvc::meshPhi(U));
    }
    vectorField surfaceVelocity(position.size(), Zero);
    forAll(surfaceVelocity, i)
    {
        surfaceVelocity[i] = tangent[i]*ut[i]
            + n[i]*meshFlux[i]/patch().magSf()[i];
    }
    vectorField::operator=(surfaceVelocity);
    fixedValueFvPatchVectorField::updateCoeffs();
}


void Foam::aotiSurfaceVelocityFvPatchVectorField::write(Ostream& os) const
{
    fvPatchVectorField::write(os);
    os.writeEntry("origin", origin_);
    os.writeEntry("axis", axis_);
    os.writeEntry("normal", normal_);
    os.writeEntry<word>("model", model_name_);
    writeEntry("value", os);
}


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{
    makePatchTypeField
    (
        fvPatchVectorField,
        aotiSurfaceVelocityFvPatchVectorField
    );
}

// ************************************************************************* //
