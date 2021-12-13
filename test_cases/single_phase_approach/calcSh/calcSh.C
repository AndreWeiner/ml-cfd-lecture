/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     |
    \\  /    A nd           | www.openfoam.com
     \\/     M anipulation  |
-------------------------------------------------------------------------------
    Copyright (C) 2011-2017 OpenFOAM Foundation
    Copyright (C) 2019 OpenCFD Ltd.
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

Application
    calcSh

Description
    Compute the normal derivative of a passive scalar at a given patch.

\*---------------------------------------------------------------------------*/

#include "fvCFD.H"
#include "mathematicalConstants.H"
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

int main(int argc, char *argv[])
{
    argList::addOption(
        "patch",
        "patchName",
        "Name of the patch");
    argList::addOption(
        "field",
        "fieldName",
        "Name of the scalar field");

#include "setRootCase.H"
#include "createTime.H"

    // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

    const word patch_name = args.getOrDefault<word>("patch", "bubble");
    const word field_name = args.getOrDefault<word>("field", "s1");
    instantList timeDirs = timeSelector::select0(runTime, args);

    scalarField globalGrad(timeDirs.size(), 0.0);
    scalarField area(timeDirs.size(), 0.0);

    forAll(timeDirs, time_i)
    {
        runTime.setTime(timeDirs[time_i], time_i);
        Info << "Time = " << runTime.timeName() << endl;

        #include "createMesh.H"

        // determine patch ID
        label surfaceID(-1);
        forAll(mesh.boundary(), patchI)
        {
            if (mesh.boundary()[patchI].name() == patch_name)
            {
                surfaceID = patchI;
            }
        }
        if (surfaceID == -1)
        {
            FatalErrorInFunction
                << "Could not find patch " << patch_name
                << abort(FatalError);
        }

        const vectorField Cf(mesh.Cf().boundaryField()[surfaceID]);
        const vectorField Sf(mesh.Sf().boundaryField()[surfaceID]);

        volScalarField sField(
            IOobject(
                field_name,
                runTime.timeName(),
                mesh,
                IOobject::MUST_READ,
                IOobject::NO_WRITE),
            mesh);

        scalarField gradS(sField.boundaryField()[surfaceID].snGrad());
        globalGrad[time_i] = sum(gradS * mag(Sf)) / sum(mag(Sf));
        area[time_i] = sum(mag(Sf));

        OFstream output_file(runTime.path() / runTime.timeName() / "localSh_" + field_name + ".csv");
        output_file.precision(15);

        output_file << "# x, y, z, area, snGrad";
        forAll(Cf, face_i)
        {
            output_file << "\n"
                        << Cf[face_i].x() << ", " << Cf[face_i].y() << ", " << Cf[face_i].z() << ", "
                        << mag(Sf[face_i]) << ", " << gradS[face_i];
        }
    } // end of time loop

    // write global gradient
    OFstream grad_file(runTime.path() / "globalSh_" + field_name + ".csv");
    grad_file.precision(15);
    grad_file << "# time, Sh_eff, area";
    forAll(timeDirs, time_i)
    {
        runTime.setTime(timeDirs[time_i], time_i);
        grad_file << "\n"
                  << runTime.timeName() << ", "
                  << globalGrad[time_i] << ", "
                  << area[time_i];
    }

    Info << "End\n"
         << endl;

    return 0;
}

// ************************************************************************* //
