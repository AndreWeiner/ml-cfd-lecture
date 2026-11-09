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

#include "aotiDisplacementPointPatchVectorField.H"
#include "pointPatchFields.H"
#include "addToRunTimeSelectionTable.H"
#include "Time.H"
#include "polyMesh.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

aotiDisplacementPointPatchVectorField::
aotiDisplacementPointPatchVectorField
(
    const pointPatch& p,
    const DimensionedField<vector, pointMesh>& iF
)
:
    fixedValuePointPatchField<vector>(p, iF),
    center_(Zero),
    direction_(Zero),
    normal_(Zero),
    radius_(0.0),
    model_name_("")
{}


aotiDisplacementPointPatchVectorField::
aotiDisplacementPointPatchVectorField
(
    const pointPatch& p,
    const DimensionedField<vector, pointMesh>& iF,
    const dictionary& dict
)
:
    fixedValuePointPatchField<vector>(p, iF, dict),
    center_(dict.lookup("center")),
    direction_(dict.lookup("direction")),
    normal_(dict.lookup("normal")),
    radius_(dict.lookupOrDefault<scalar>("radius", 0.0)),
    model_name_(dict.get<word>("model")),
    shape_model_(std::make_shared<AotiModel>(model_name_.c_str()))
{
    if (!dict.found("value"))
    {
        updateCoeffs();
    }
}


aotiDisplacementPointPatchVectorField::
aotiDisplacementPointPatchVectorField
(
    const aotiDisplacementPointPatchVectorField& ptf,
    const pointPatch& p,
    const DimensionedField<vector, pointMesh>& iF,
    const pointPatchFieldMapper& mapper
)
:
    fixedValuePointPatchField<vector>(ptf, p, iF, mapper),
    center_(ptf.center_),
    direction_(ptf.direction_),
    normal_(ptf.normal_),
    radius_(ptf.radius_),
    model_name_(ptf.model_name_),
    shape_model_(ptf.shape_model_)
{}


aotiDisplacementPointPatchVectorField::
aotiDisplacementPointPatchVectorField
(
    const aotiDisplacementPointPatchVectorField& ptf,
    const DimensionedField<vector, pointMesh>& iF
)
:
    fixedValuePointPatchField<vector>(ptf, iF),
    center_(ptf.center_),
    direction_(ptf.direction_),
    normal_(ptf.normal_),
    radius_(ptf.radius_),
    model_name_(ptf.model_name_),
    shape_model_(ptf.shape_model_)
{}


// * * * * * * * * * * * * * * * Member Functions  * * * * * * * * * * * * * //

void aotiDisplacementPointPatchVectorField::updateCoeffs()
{
    if (this->updated())
    {
        return;
    }

    const pointField& points = patch().localPoints();
    std::vector<double> features(2*points.size());
    vectorField radialDirection(points.size(), Zero);
    forAll(points, i)
    {
        const vector relative = points[i] - center_;
        const vector planar = relative - normal_*(relative & normal_);
        const scalar axial = planar & direction_;
        features[2*i] = db().time().value();
        features[2*i+1] = atan2(mag(planar-axial*direction_), axial);
        radialDirection[i] = relative/mag(relative);
    }
    const auto radius = shape_model_->predict(features, points.size(), 2);
    vectorField result(points.size(), Zero);
    forAll(result, i)
    {
        result[i] = radialDirection[i]*(radius[i]-radius_);
    }
    Field<vector>::operator=(result);

    fixedValuePointPatchField<vector>::updateCoeffs();
}


void aotiDisplacementPointPatchVectorField::write(Ostream& os) const
{
    pointPatchField<vector>::write(os);
    os.writeEntry("center", center_);
    os.writeEntry("direction", direction_);
    os.writeEntry("normal", normal_);
    os.writeEntry("radius", radius_);
    os.writeEntry<word>("model", model_name_);
    writeEntry("value", os);
}


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

makePointPatchTypeField
(
    pointPatchVectorField,
    aotiDisplacementPointPatchVectorField
);

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// ************************************************************************* //
