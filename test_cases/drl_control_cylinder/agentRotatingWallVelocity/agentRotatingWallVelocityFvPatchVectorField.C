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

\*---------------------------------------------------------------------------*/

#include "agentRotatingWallVelocityFvPatchVectorField.H"
#include "addToRunTimeSelectionTable.H"
#include "volFields.H"
#include "surfaceFields.H"

// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

Foam::agentRotatingWallVelocityFvPatchVectorField::
    agentRotatingWallVelocityFvPatchVectorField(
        const fvPatch &p,
        const DimensionedField<vector, volMesh> &iF)
    : fixedValueFvPatchField<vector>(p, iF),
      origin_(),
      axis_(Zero)
{
}

Foam::agentRotatingWallVelocityFvPatchVectorField::
    agentRotatingWallVelocityFvPatchVectorField(
        const fvPatch &p,
        const DimensionedField<vector, volMesh> &iF,
        const dictionary &dict)
    : fixedValueFvPatchField<vector>(p, iF, dict, false),
      origin_(dict.get<vector>("origin")),
      axis_(dict.get<vector>("axis")),
      train_(dict.get<bool>("train")),
      interval_(dict.get<int>("interval")),
      start_time_(dict.get<scalar>("startTime")),
      start_iter_(0),
      policy_name_(dict.get<word>("policy")),
      policy_(torch::jit::load(policy_name_)),
      abs_omega_max_(dict.get<scalar>("absOmegaMax")),
      log_std_max_(dict.get<scalar>("logStdMax")),
      omega_(0.0),
      omega_old_(0.0),
      control_time_(0.0),
      theta_cumulative_(0.0),
      dt_theta_cumulative_(0.0)
{
    if (dict.found("value"))
    {
        fvPatchField<vector>::operator=(
            vectorField("value", dict, p.size()));
    }
    else
    {
        // Evaluate the wall velocity
        updateCoeffs();
    }
}

Foam::agentRotatingWallVelocityFvPatchVectorField::
    agentRotatingWallVelocityFvPatchVectorField(
        const agentRotatingWallVelocityFvPatchVectorField &ptf,
        const fvPatch &p,
        const DimensionedField<vector, volMesh> &iF,
        const fvPatchFieldMapper &mapper)
    : fixedValueFvPatchField<vector>(ptf, p, iF, mapper),
      origin_(ptf.origin_),
      axis_(ptf.axis_),
      train_(ptf.train_),
      interval_(ptf.interval_),
      start_time_(ptf.start_time_),
      start_iter_(ptf.start_iter_),
      policy_name_(ptf.policy_name_),
      policy_(ptf.policy_),
      abs_omega_max_(ptf.abs_omega_max_),
      log_std_max_(ptf.log_std_max_),
      omega_(ptf.omega_),
      omega_old_(ptf.omega_old_),
      control_time_(ptf.control_time_),
      theta_cumulative_(ptf.theta_cumulative_),
      dt_theta_cumulative_(ptf.dt_theta_cumulative_)
{
}

Foam::agentRotatingWallVelocityFvPatchVectorField::
    agentRotatingWallVelocityFvPatchVectorField(
        const agentRotatingWallVelocityFvPatchVectorField &rwvpvf)
    : fixedValueFvPatchField<vector>(rwvpvf),
      origin_(rwvpvf.origin_),
      axis_(rwvpvf.axis_),
      train_(rwvpvf.train_),
      interval_(rwvpvf.interval_),
      start_time_(rwvpvf.start_time_),
      start_iter_(rwvpvf.start_iter_),
      policy_name_(rwvpvf.policy_name_),
      policy_(rwvpvf.policy_),
      abs_omega_max_(rwvpvf.abs_omega_max_),
      log_std_max_(rwvpvf.log_std_max_),
      omega_(rwvpvf.omega_),
      omega_old_(rwvpvf.omega_old_),
      control_time_(rwvpvf.control_time_),
      theta_cumulative_(rwvpvf.theta_cumulative_),
      dt_theta_cumulative_(rwvpvf.dt_theta_cumulative_)
{
}

Foam::agentRotatingWallVelocityFvPatchVectorField::
    agentRotatingWallVelocityFvPatchVectorField(
        const agentRotatingWallVelocityFvPatchVectorField &rwvpvf,
        const DimensionedField<vector, volMesh> &iF)
    : fixedValueFvPatchField<vector>(rwvpvf, iF),
      origin_(rwvpvf.origin_),
      axis_(rwvpvf.axis_),
      train_(rwvpvf.train_),
      interval_(rwvpvf.interval_),
      start_time_(rwvpvf.start_time_),
      start_iter_(rwvpvf.start_iter_),
      policy_name_(rwvpvf.policy_name_),
      policy_(rwvpvf.policy_),
      abs_omega_max_(rwvpvf.abs_omega_max_),
      log_std_max_(rwvpvf.log_std_max_),
      omega_(rwvpvf.omega_),
      omega_old_(rwvpvf.omega_old_),
      control_time_(rwvpvf.control_time_),
      theta_cumulative_(rwvpvf.theta_cumulative_),
      dt_theta_cumulative_(rwvpvf.dt_theta_cumulative_)
{
}

// * * * * * * * * * * * * * * * Member Functions  * * * * * * * * * * * * * //

void Foam::agentRotatingWallVelocityFvPatchVectorField::updateCoeffs()
{
    if (updated())
    {
        return;
    }

    // update angular velocity
    const scalar t = this->db().time().timeOutputValue();
    bool steps_remaining = (this->db().time().timeIndex() - start_iter_) % interval_ == 0;
    if (t >= start_time_)
    {
        if(start_iter_ == 0)
        {
            start_iter_ = this->db().time().timeIndex();
            steps_remaining = true;
        }
        if (steps_remaining && update_omega_)
        {
            Info << "Updating Omega with policy.\n";
            omega_old_ = omega_;
            control_time_ = t;

            const fvPatchField<scalar> &p = patch().lookupPatchField<volScalarField, scalar>("p");
            // Create lists of the variables on each processor so that they can be gathered onto the master processor later.
            List<scalar> pList(p.size());

            // Populate the above lists.
            forAll(p, i)
            {
                pList[i] = p[i];
            }

            // Create lists of the lists of the above variables, with size equal to the
            // number of processors.
            List< List<scalar> > gatheredValues(Pstream::nProcs());

            //  Populate and gather the stuff onto the master processor.
            gatheredValues[Pstream::myProcNo()] = pList;
            Pstream::gatherList(gatheredValues);

            if (Pstream::master()) //only run on the master
            {
                // creating the feature vector
                int size = 0;
                for (int i = 0; i < gatheredValues.size(); i++)
                {
                    size += gatheredValues[i].size();
                }
                torch::Tensor features = torch::zeros({ 1, size }, torch::kFloat64);
                int k = 0;
                std::vector<scalar> pvec(size);
                for (int i = 0; i < gatheredValues.size(); i++)
                {
                    for (int j = 0; j < gatheredValues[i].size(); j++)
                    {
                        features[0][k] = gatheredValues[i][j];
                        pvec[k] = gatheredValues[i][j];
                        k++;
                    }
                }
                std::vector<torch::jit::IValue> policyFeatures{features};
                torch::Tensor dist_parameters = policy_.forward(policyFeatures).toTensor();
                scalar alpha = dist_parameters[0][0].item<double>();
                scalar beta = dist_parameters[0][1].item<double>();
                std::random_device rd;
                std::mt19937 gen(rd());
                std::gamma_distribution<double> distribution_1(alpha, 1.0);
                std::gamma_distribution<double> distribution_2(beta, 1.0);
                scalar omega_pre_scale;
                if (train_)
                {
                    // sample from Beta distribution during training
                    double number_1 = distribution_1(gen);
                    double number_2 = distribution_2(gen);
                    omega_pre_scale = number_1 / (number_1 + number_2);
                }
                else
                {
                    // use expected (mean) angular velocity
                    omega_pre_scale = alpha / (alpha + beta);
                }
                // rescale to actionspace
                omega_ = (omega_pre_scale - 0.5) * 2 * abs_omega_max_;
                // save trajectory
                scalar mean = (alpha / (alpha + beta) - 0.5) * 2 * abs_omega_max_;
                scalar log_std = sqrt((alpha * beta)/ ((alpha + beta + 1) * (alpha + beta) * (alpha + beta)));
                //not yet implemented
                scalar entropy = 0;
                scalar log_p = log(omega_pre_scale) * (alpha - 1.0) + log(1 - omega_pre_scale) * (beta - 1.0) + std::lgamma(alpha + beta) - (std::lgamma(alpha) + std::lgamma(beta));
                saveTrajectory(log_p, entropy, mean, log_std, alpha, beta, pvec, size);
                // reset cumulative values
                theta_cumulative_ = 0.0;
                dt_theta_cumulative_ = 0.0;
                Info << "New omega: " << omega_ << "; old value: " << omega_old_ << "\n";
            }
            Pstream::scatter(omega_);

            // avoid update of angular velocity during p-U coupling
            update_omega_ = false;
        }
    }

    // activate update of angular velocity after p-U coupling
    if (!steps_remaining)
    {
        update_omega_ = true;
    }

    // update angular velocity by linear transition from old to new value
    const scalar dt = this->db().time().deltaTValue();
    scalar d_omega = (omega_ - omega_old_) / (dt * interval_) * (t - control_time_);
    scalar omega = omega_old_ + d_omega;
    theta_cumulative_ += abs(omega) * dt;
    dt_theta_cumulative_ += abs(omega);

    // Calculate the rotating wall velocity from the specification of the motion

    const vectorField Up(
        (-omega) * ((patch().Cf() - origin_) ^ (axis_ / mag(axis_))));

    // Remove the component of Up normal to the wall
    // just in case it is not exactly circular

    const vectorField n(patch().nf());
    vectorField::operator=(Up - n * (n & Up));

    fixedValueFvPatchVectorField::updateCoeffs();
}

void Foam::agentRotatingWallVelocityFvPatchVectorField::write(Ostream &os) const
{
    fvPatchVectorField::write(os);
    os.writeEntry("origin", origin_);
    os.writeEntry("axis", axis_);
    os.writeEntry("policy", policy_name_);
    os.writeEntry("startTime", start_time_);
    os.writeEntry("interval", interval_);
    os.writeEntry("train", train_);
    os.writeEntry("absOmegaMax", abs_omega_max_);
    os.writeEntry("logStdMax", log_std_max_);
}

void Foam::agentRotatingWallVelocityFvPatchVectorField::saveTrajectory(scalar log_p, scalar entropy, scalar mean, scalar log_std, scalar alpha, scalar beta, std::vector<scalar> pvec, int size) const
{
    std::ifstream file("trajectory.csv");
    std::fstream trajectory("trajectory.csv", std::ios::app | std::ios::binary);
    const scalar t = this->db().time().timeOutputValue();
    if(!file.good())
    {
        // write header
        trajectory << "t, omega, omega_mean, omega_log_std, alpha, beta, log_prob, entropy, theta_sum, dt_theta_sum, p(" << size << ")";
    }
    trajectory << std::setprecision(15)
               << "\n"
               << t << ", "
               << omega_ << ", "
               << mean << ", "
               << log_std << ", "
               << alpha << ", "
               << beta << ", "
               << log_p << ", "
               << entropy << ", "
               << theta_cumulative_ << ", "
               << dt_theta_cumulative_;
    
    for (int i = 0; i < size; i++)
    {
        trajectory << ", " << pvec[i];
    }
}

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{
    makePatchTypeField(
        fvPatchVectorField,
        agentRotatingWallVelocityFvPatchVectorField);
}

// ************************************************************************* //
