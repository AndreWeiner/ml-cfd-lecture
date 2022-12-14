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

#include "ptDisplacementPointPatchVectorField.H"
#include "pointPatchFields.H"
#include "addToRunTimeSelectionTable.H"
#include "Time.H"
#include "polyMesh.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

ptDisplacementPointPatchVectorField::
ptDisplacementPointPatchVectorField
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


ptDisplacementPointPatchVectorField::
ptDisplacementPointPatchVectorField
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
    model_name_(dict.get<word>("modelName")),
    shape_model_(torch::jit::load(model_name_))
{
    if (!dict.found("value"))
    {
        updateCoeffs();
    }
}


ptDisplacementPointPatchVectorField::
ptDisplacementPointPatchVectorField
(
    const ptDisplacementPointPatchVectorField& ptf,
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


ptDisplacementPointPatchVectorField::
ptDisplacementPointPatchVectorField
(
    const ptDisplacementPointPatchVectorField& ptf,
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

void ptDisplacementPointPatchVectorField::updateCoeffs()
{
    if (this->updated())
    {
        return;
    }

    const polyMesh& mesh = this->internalField().mesh()();
    const Time& t = mesh.time();
    const pointField& localPoints = patch().localPoints();
    torch::Tensor featureTensor = torch::ones({localPoints.size(), 2}, torch::kFloat64);
    const vector xy = vector{1, 1, 1} - normal_;
    const vector unit_x = xy - direction_;

    forAll(localPoints, i)
    {
        vector x = (localPoints[i] & unit_x) * unit_x + (localPoints[i] & direction_) * direction_  - center_;
        scalar r = sqrt(x & x);
        scalar theta = acos((x & direction_) / r);
        featureTensor[i][0] = theta;
        featureTensor[i][1] = t.value();
    }

    std::vector<torch::jit::IValue> features{featureTensor};
    torch::Tensor radTensor = shape_model_.forward(features).toTensor();
    auto radAccessor = radTensor.accessor<double,1>();
    vectorField result(localPoints.size(), Zero);

    forAll(result, i)
    {
        vector rad = localPoints[i] - center_;
        result[i] = rad / mag(rad) * (radAccessor[i]-radius_);
    }

    Field<vector>::operator=(result);

    fixedValuePointPatchField<vector>::updateCoeffs();
}


void ptDisplacementPointPatchVectorField::write(Ostream& os) const
{
    pointPatchField<vector>::write(os);
    os.writeEntry("center", center_);
    os.writeEntry("direction", direction_);
    os.writeEntry("normal", normal_);
    os.writeEntry("radius", radius_);
    os.writeEntry<word>("modelName", model_name_);
    writeEntry("value", os);
}


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

makePointPatchTypeField
(
    pointPatchVectorField,
    ptDisplacementPointPatchVectorField
);

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// ************************************************************************* //
