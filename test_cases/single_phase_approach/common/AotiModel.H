/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     | Website: https://www.openfoam.com
    \\  /    A nd           |
     \\/     M anipulation  |
-------------------------------------------------------------------------------
Class
    Foam::AotiModel
Description
    CPU float64 AOTInductor inference for the course bubble models. One package is
    shared by mapped/cloned patch fields; one batched call is made per patch.
    The graph owns preprocessing and accepts features[N,K], prediction[N,1].
    SPDX-License-Identifier: GPL-3.0-or-later
\*---------------------------------------------------------------------------*/
#ifndef courseAotiModel_H
#define courseAotiModel_H
#include <ATen/ATen.h>
#include <torch/csrc/inductor/aoti_package/model_package_loader.h>
#include <vector>
#include <stdexcept>
namespace Foam
{
class AotiModel
{
    mutable torch::inductor::AOTIModelPackageLoader model_;
public:
    explicit AotiModel(const char* path) : model_(path, "model", true) {}
    std::vector<double> predict
    (
        const std::vector<double>& features, int64_t rows, int64_t columns
    ) const
    {
        if (!rows) return {};
        const auto input = at::from_blob
        (
            const_cast<double*>(features.data()), {rows, columns},
            at::TensorOptions().dtype(at::kDouble).device(at::kCPU)
        );
        const auto values = model_.run({input});
        if (values.size() != 1 || values[0].scalar_type() != at::kDouble
         || values[0].sizes() != at::IntArrayRef({rows, 1}))
        {
            throw std::runtime_error("Expected float64 prediction[N,1]");
        }
        const auto output = values[0].contiguous();
        const double* result = output.data_ptr<double>();
        return {result, result+rows};
    }
};
}
#endif
// ************************************************************************* //
