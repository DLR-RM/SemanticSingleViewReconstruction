#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"

using namespace tensorflow;

REGISTER_OP("ReachableSpaceDetect")
    .Input("loss_tensor: uint8")
    .Output("loss: float32");

#include "tensorflow/core/framework/op_kernel.h"
#include "LossValueType.h"

using namespace tensorflow;

class ReachableSpaceDetectOp : public OpKernel {
 public:
  explicit ReachableSpaceDetectOp(OpKernelConstruction* context) : OpKernel(context) {}

  float extractInputScalar(OpKernelContext* context, const int id, const std::string& name){
    const Tensor& tensor = context->input(id);
    return tensor.scalar<float>().data()[0];
  }

  void Compute(OpKernelContext* context) override {
    // Grab the input tensor
    const Tensor& input_tensor = context->input(0);
    auto input = input_tensor.tensor<uint8, 3>();

    const int usedInputSize = 256;
    const int usedOutputSize = 16;
    constexpr const int blockSize = usedInputSize / usedOutputSize;
    const int amountOfClasses = 10;

    const int lowerClassEndNumber = LossValueType::getClassValueTrueWall(0);
    const int upperClassEndNumber = LossValueType::getClassValueNotVisible(amountOfClasses, amountOfClasses);

    // Create an output tensor
    Tensor* output_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0.0, {usedOutputSize, usedOutputSize, usedOutputSize},
                                                     &output_tensor));
    auto output = output_tensor->tensor<float, 3>();
    for(int x = 0; x < usedOutputSize; ++x){
        for(int y = 0; y < usedOutputSize; ++y){
            for(int z = 0; z < usedOutputSize; ++z){
                float mean_value = 0.0;
                float counter = 1.0;
				for(unsigned int i = 0; i < blockSize; ++i){
					for(unsigned int j = 0; j < blockSize; ++j){
						for(unsigned int k = 0; k < blockSize; ++k){
                            const float fac = 1.0 / counter;
                            float isReachable = 1.0;
                            const unsigned char currentValue = input(x * blockSize + i, y * blockSize + j, z * blockSize + k);
                            if(currentValue == LossValueType::notReachable() || currentValue == LossValueType::wallValueNotVisible() ||
                                currentValue == LossValueType::trueWallValueNotVisible() ||
                                (currentValue >= LossValueType::freeNotVisibleStart() && currentValue <= LossValueType::freeNotVisibleEnd()) ||
                                (currentValue >= LossValueType::getClassValueNotVisibleTrueWall(0, amountOfClasses) && currentValue < LossValueType::getClassValueNotVisibleTrueWall(amountOfClasses, amountOfClasses)) ||
                                (currentValue >= LossValueType::getClassValueNotVisible(0, amountOfClasses) && currentValue < LossValueType::getClassValueNotVisible(amountOfClasses, amountOfClasses))
                                ){
                                isReachable = 0.0;
                            }else/** these are all else: if (currentValue == LossValueType::freeValue() || currentValue == LossValueType::wallValue() || currentValue == LossValueType::trueWallValue() ||
                                (currentValue >= LossValueType::getClassValueTrueWall(0) && currentValue < LossValueType::getClassValueTrueWall(amountOfClasses)) ||
                                (currentValue >= LossValueType::getClassValue(0, amountOfClasses) && currentValue < LossValueType::getClassValue(amountOfClasses, amountOfClasses))
                            )**/ {
                                isReachable = 1.0;
                            }
                            mean_value = fac * float(isReachable) + (1.0 - fac) * mean_value;
                            counter += 1;
                        }
                    }
                }
				output(x,y,z) = mean_value;
            }
        }
    }
  }
};


REGISTER_KERNEL_BUILDER(Name("ReachableSpaceDetect").Device(DEVICE_CPU), ReachableSpaceDetectOp);