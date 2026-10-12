/*---------------------------------------------------------------------------*\
Application
    tensorCreation

Description
    Application to demonstrate how to compile PyTorch C++ source code with
    wmake, e.g., to use PyTorch models in a flow solver or boundary condition.
    Several examples for basic working with PyTorch tensors are also included:
    - tensor creation with linspace and ones
    - basic mathematical functions
    - autograd to compute derivatives
    - saving tensors to disk
    - creating at::Tensor based on std::vector
\*---------------------------------------------------------------------------*/

// Include LibTorch before OpenFOAM defines its TypeName macro.
#include <torch/torch.h>
#include "argList.H"
#include <iomanip>
#include <iostream>
#include <vector>

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

using std::cout;
using std::setw;
using std::vector;

int main(int argc, char *argv[])
{
    // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

    auto options = torch::TensorOptions()
        .dtype(torch::kFloat64)
        .requires_grad(true);

    cout << "->Creating a tensor with 10 values linearly spaced between 0 and 2xPi:\n";
    int64_t n_val = 10;
    torch::Tensor x = torch::linspace(0.0, 6.28, n_val, options);
    cout << "Created tensor of shape " << x.sizes() << "\n";
    cout << x << "\n";

    cout << "\n->Computing sin(x):\n";
    auto sin_x = torch::sin(x);
    cout << sin_x << "\n";

    cout << "\n->Computing d/dx sin(x):\n";
    sin_x.backward(torch::ones(n_val));
    auto dx_sin_x = x.grad();
    cout << dx_sin_x << "\n";

    cout << "\n->Comparing autograd and analytical solution:\n";
    auto analytical = torch::cos(x);
    TORCH_CHECK(torch::allclose(dx_sin_x, analytical, 1e-10, 1e-12), "Autograd derivative does not match cos(x)");
    auto analytical_accessor = analytical.accessor<double,1>();
    auto autograd = dx_sin_x.accessor<double,1>();
    cout << setw(15) << "autograd" << setw(15) << "analytical\n";
    for (int64_t i=0; i < n_val; i++) {
        cout << setw(15) << autograd[i] << setw(15) << analytical_accessor[i] << "\n";
    }

    cout << "\n->Saving the derivative for later use...\n";
    torch::save(dx_sin_x, "derivative.pt");

    cout << "\n->Initializing a at::Tensor based on a std::vector:\n";
    vector<double> some_vector{1.0, 2.0, 3.0};
    auto tensor_from_vector = torch::from_blob(some_vector.data(), some_vector.size(), options).clone();
    cout << tensor_from_vector << "\n";
    TORCH_CHECK(torch::allclose(tensor_from_vector, torch::tensor({1.0, 2.0, 3.0}, torch::kFloat64)), "Tensor values differ");
    torch::Tensor loaded;
    torch::load(loaded, "derivative.pt");
    TORCH_CHECK(torch::allclose(loaded, dx_sin_x), "Saved tensor differs after loading");


    cout << "\n->The end\n";

    return 0;
}

// ************************************************************************* //
