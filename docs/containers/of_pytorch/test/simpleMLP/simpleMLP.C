/* Example showing how to compile a PyTorch C++ application using CMake.
See https://pytorch.org/tutorials/advanced/cpp_frontend.html for more
examples and detailed documentation. */

#include <torch/torch.h>
#include <iostream>

using torch::nn::Module;
using torch::Tensor;
using torch::nn::Linear;
using torch::sigmoid;
using std::cout;

class SimpleMLPImpl : public Module {
  // Implementation of a simple multilayer preceptron with a single hidden layer
  public:
    SimpleMLPImpl(const int64_t input_size, const int64_t hidden_size, const int64_t output_size)
        : _linear_1(Linear(input_size, hidden_size)),
          _linear_2(Linear(hidden_size, output_size)) {
            register_module("linear_1", _linear_1);
            register_module("linear_2", _linear_2);
          }

    Tensor forward(Tensor input) {
      input = sigmoid(_linear_1->forward(input));
      return _linear_2->forward(input);
    }
  private:
    Linear _linear_1;
    Linear _linear_2;
};

TORCH_MODULE(SimpleMLP);

int main() {
    torch::manual_seed(42);
    Tensor test_input = torch::ones({5, 2});
    cout << "Shape of input tensor:\n" << test_input.sizes() << "\n";
    auto model = SimpleMLP(2, 10, 1);
    Tensor result = model->forward(test_input);
    cout << "Shape of output tensor:\n" << result.sizes() << "\n";
    TORCH_CHECK(result.sizes() == torch::IntArrayRef({5, 1}), "Unexpected MLP output shape");
    TORCH_CHECK(torch::isfinite(result).all().item<bool>(), "MLP produced non-finite values");
    return 0;
}