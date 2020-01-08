/*
 *  MIT License
 *
 *  Copyright (c) 2017 Sadeep Jayasumana
 *
 *  Permission is hereby granted, free of charge, to any person obtaining a copy
 *  of this software and associated documentation files (the "Software"), to deal
 *  in the Software without restriction, including without limitation the rights
 *  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 *  copies of the Software, and to permit persons to whom the Software is
 *  furnished to do so, subject to the following conditions:

 *  The above copyright notice and this permission notice shall be included in all
 *  copies or substantial portions of the Software.

 *  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 *  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 *  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 *  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 *  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 *  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 *  SOFTWARE.
 */

// #include "tensorflow/core/framework/op.h"
// #include "tensorflow/core/framework/shape_inference.h"
// #include "tensorflow/core/framework/op_kernel.h"
// #include "tensorflow/core/framework/tensor_shape.h"
// #include "modified_permutohedral.h"

namespace mxnet{


namespace mxnet {
namespace op {

DMLC_REGISTER_PARAMETER(DiagParam);

NNVM_REGISTER_OP(diag)
.describe(R"code(Extracts a diagonal or constructs a diagonal array.
``diag``'s behavior depends on the input array dimensions:
- 1-D arrays: constructs a 2-D array with the input as its diagonal, all other elements are zero.
- N-D arrays: extracts the diagonals of the sub-arrays with axes specified by ``axis1`` and ``axis2``.
  The output shape would be decided by removing the axes numbered ``axis1`` and ``axis2`` from the
  input shape and appending to the result a new axis with the size of the diagonals in question.
  For example, when the input shape is `(2, 3, 4, 5)`, ``axis1`` and ``axis2`` are 0 and 2
  respectively and ``k`` is 0, the resulting shape would be `(3, 5, 2)`.
Examples::
  x = [[1, 2, 3],
       [4, 5, 6]]
  diag(x) = [1, 5]
  diag(x, k=1) = [2, 6]
  diag(x, k=-1) = [4]
  x = [1, 2, 3]
  diag(x) = [[1, 0, 0],
             [0, 2, 0],
             [0, 0, 3]]
  diag(x, k=1) = [[0, 1, 0],
                  [0, 0, 2],
                  [0, 0, 0]]
  diag(x, k=-1) = [[0, 0, 0],
                   [1, 0, 0],
                   [0, 2, 0]]
  x = [[[1, 2],
        [3, 4]],
       [[5, 6],
        [7, 8]]]
  diag(x) = [[1, 7],
             [2, 8]]
  diag(x, k=1) = [[3],
                  [4]]
  diag(x, axis1=-2, axis2=-1) = [[1, 4],
                                 [5, 8]]
)code" ADD_FILELINE)
.set_attr_parser(ParamParser<DiagParam>)
.set_num_inputs(1)
.set_num_outputs(1)
.set_attr<nnvm::FListInputNames>("FListInputNames",
  [](const NodeAttrs& attrs) {
    return std::vector<std::string>{"data"};
  })
.set_attr<mxnet::FInferShape>("FInferShape", DiagOpShape)
.set_attr<nnvm::FInferType>("FInferType", DiagOpType)
.set_attr<FCompute>("FCompute<cpu>", DiagOpForward<cpu>)
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseNone{"_backward_diag"})
.add_argument("data", "NDArray-or-Symbol", "Input ndarray")
.add_arguments(DiagParam::__FIELDS__());


NNVM_REGISTER_OP(_backward_diag)
.set_attr_parser(ParamParser<DiagParam>)
.set_num_inputs(1)
.set_num_outputs(1)
.set_attr<nnvm::TIsBackward>("TIsBackward", true)
.set_attr<FCompute>("FCompute<cpu>", DiagOpBackward<cpu>);


}  // namespace op
}  // namespace mxnet