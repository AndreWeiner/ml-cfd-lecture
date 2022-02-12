/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     |
    \\  /    A nd           | www.openfoam.com
     \\/     M anipulation  |
-------------------------------------------------------------------------------
    Copyright (C) 2011-2016 OpenFOAM Foundation
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
    Foam::agentRotatingWallVelocityFvPatchVectorField

Group
    grpWallBoundaryConditions grpGenericBoundaryConditions

Description
    This boundary condition provides a rotational velocity condition.

Usage
    \table
        Property     | Description             | Required    | Default value
        origin       | origin of rotation in Cartesian coordinates | yes|
        axis         | axis of rotation        | yes         |
    \endtable

    Example of the boundary condition specification:
    \verbatim
    <patchName>
    {
        type            agentRotatingWallVelocity;
        origin          (0 0 0);
        axis            (0 0 1);
        policy          "policy.pt"
        startTime       0.01;
        interval        30;
        train           true;
        absOmegaMax     0.05;
        logStdMax       2.0;
    }
    \endverbatim

    The \c omega entry is a Function1 of time, see Foam::Function1Types.

See also
    Foam::fixedValueFvPatchField
    Foam::Function1Types

SourceFiles
    agentRotatingWallVelocityFvPatchVectorField.C

\*---------------------------------------------------------------------------*/

#ifndef agentRotatingWallVelocityFvPatchVectorField_H
#define agentRotatingWallVelocityFvPatchVectorField_H

#include <torch/script.h>
#include "fixedValueFvPatchFields.H"
#include <random>

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

/*---------------------------------------------------------------------------*\
           Class agentRotatingWallVelocityFvPatchVectorField Declaration
\*---------------------------------------------------------------------------*/

class agentRotatingWallVelocityFvPatchVectorField
:
    public fixedValueFvPatchVectorField
{
    // Private data

        //- origin of the rotation
        vector origin_;

        //- axis of the rotation
        vector axis_;

        //- training or evaluation mode
        bool train_;

        //- number of iterations between updates
        int interval_;

        //- when to start controlling
        scalar start_time_;

        //- time iteration at which start time was reached
        int start_iter_;

        //- name of the PyTorch angular velocity model
        word policy_name_;

        //- PyTorch model predicting mean and log(std) of angular velocity
        torch::jit::script::Module policy_;

        //- largest allowed absolute value of angular velocity
        scalar abs_omega_max_;

        //- maximum allowed logarithmic standard deviation
        scalar log_std_max_;

        //- next predicted angular velocity
        scalar omega_;

        //- previously predicted angular velocity
        scalar omega_old_;

        //- last time at which angular velocity was computed
        scalar control_time_;

        //- cumulative angular motion
        scalar theta_cumulative_;

        //- cumulative time derivative of angular motion
        scalar dt_theta_cumulative_;

        //- update of next angular velocity value if true
        bool update_omega_;


public:

    //- Runtime type information
    TypeName("agentRotatingWallVelocity");


    // Constructors

        //- Construct from patch and internal field
        agentRotatingWallVelocityFvPatchVectorField
        (
            const fvPatch&,
            const DimensionedField<vector, volMesh>&
        );

        //- Construct from patch, internal field and dictionary
        agentRotatingWallVelocityFvPatchVectorField
        (
            const fvPatch&,
            const DimensionedField<vector, volMesh>&,
            const dictionary&
        );

        //- Construct by mapping given agentRotatingWallVelocityFvPatchVectorField
        //  onto a new patch
        agentRotatingWallVelocityFvPatchVectorField
        (
            const agentRotatingWallVelocityFvPatchVectorField&,
            const fvPatch&,
            const DimensionedField<vector, volMesh>&,
            const fvPatchFieldMapper&
        );

        //- Construct as copy
        agentRotatingWallVelocityFvPatchVectorField
        (
            const agentRotatingWallVelocityFvPatchVectorField&
        );

        //- Construct and return a clone
        virtual tmp<fvPatchVectorField> clone() const
        {
            return tmp<fvPatchVectorField>
            (
                new agentRotatingWallVelocityFvPatchVectorField(*this)
            );
        }

        //- Construct as copy setting internal field reference
        agentRotatingWallVelocityFvPatchVectorField
        (
            const agentRotatingWallVelocityFvPatchVectorField&,
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
                new agentRotatingWallVelocityFvPatchVectorField(*this, iF)
            );
        }



    // Member functions

        //- Update the coefficients associated with the patch field
        virtual void updateCoeffs();

        //- Write
        virtual void write(Ostream&) const;

        //- Save trajectory to file
        void saveTrajectory(scalar log_p, scalar entropy, scalar mean, scalar log_std, scalar alpha, scalar beta, std::vector<scalar> pvec, int size) const;
};


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif

// ************************************************************************* //
