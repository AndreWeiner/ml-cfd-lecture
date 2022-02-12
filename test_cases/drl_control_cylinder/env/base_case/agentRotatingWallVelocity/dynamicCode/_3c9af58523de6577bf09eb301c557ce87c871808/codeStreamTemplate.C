/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     |
    \\  /    A nd           | www.openfoam.com
     \\/     M anipulation  |
-------------------------------------------------------------------------------
    Copyright (C) YEAR AUTHOR, AFFILIATION
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

Description
    Template for use with codeStream.

\*---------------------------------------------------------------------------*/

#include "dictionary.H"
#include "Ostream.H"
#include "Pstream.H"
#include "pointField.H"
#include "tensor.H"
#include "unitConversion.H"

//{{{ begin codeInclude
#line 35 "/home/andre/Development/ml-cfd-lecture/test_cases/drl_control_cylinder/env/base_case/agentRotatingWallVelocity/system/blockMeshDict.#codeStream"
#include "pointField.H"
//}}} end codeInclude

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

// * * * * * * * * * * * * * * * Local Functions * * * * * * * * * * * * * * //

//{{{ begin localCode

//}}} end localCode


// * * * * * * * * * * * * * * * Global Functions  * * * * * * * * * * * * * //

extern "C" void codeStream_3c9af58523de6577bf09eb301c557ce87c871808(Foam::Ostream& os, const Foam::dictionary& dict)
{
//{{{ begin code
    #line 40 "/home/andre/Development/ml-cfd-lecture/test_cases/drl_control_cylinder/env/base_case/agentRotatingWallVelocity/system/blockMeshDict.#codeStream"
pointField points({
            /* 0*/ {0, 0, 0},
            /* 1*/ {0.20000000 * 2, 0, 0},
            /* 2*/ {2.20000000, 0, 0},
            /* 3*/ {2.20000000, 0.41000000, 0},
            /* 4*/ {0.20000000 * 2, 0.41000000, 0},
            /* 5*/ {0, 0.41000000, 0},
            /* 6*/ {0.20000000 - 0.03535534, 0.20000000 - 0.03535534, 0},
            /* 7*/ {0.20000000 + 0.03535534, 0.20000000 - 0.03535534, 0},
            /* 8*/ {0.20000000 - 0.03535534, 0.20000000 + 0.03535534, 0},
            /* 9*/ {0.20000000 + 0.03535534, 0.20000000 + 0.03535534, 0}
        });

        // Duplicate z points for thickness
        const label sz = points.size();
        points.resize(2*sz);
        for (label i = 0; i < sz; ++i)
        {
            const point& pt = points[i];
            points[i + sz] = point(pt.x(), pt.y(), 0.01000000);
        }

        os  << points;
//}}} end code
}


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// ************************************************************************* //

