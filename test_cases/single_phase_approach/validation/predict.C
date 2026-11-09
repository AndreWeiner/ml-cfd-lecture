/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     | Website: https://www.openfoam.com
    \\  /    A nd           |
     \\/     M anipulation  |
-------------------------------------------------------------------------------
Application
    predict
Description
    Validate the same AOTInductor helper used by the boundary conditions against
    Python. Read row-major float64 features from stdin; write one value per row.
    SPDX-License-Identifier: GPL-3.0-or-later
\*---------------------------------------------------------------------------*/
#include "AotiModel.H"
#include <iostream>
#include <iomanip>
#include <cstdlib>
int main(int argc, char** argv)
{
    if (argc != 3) return 1;
    const int columns = std::atoi(argv[2]);
    std::vector<double> input;
    double value;
    while (std::cin >> value) input.push_back(value);
    Foam::AotiModel model(argv[1]);
    auto output = model.predict(input, input.size()/columns, columns);
    for (double item : output) std::cout << std::setprecision(17) << item << '\n';
}
// ************************************************************************* //
