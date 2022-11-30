#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"

using namespace tensorflow;

REGISTER_OP("LossConversion")
    .Input("loss_tensor: uint8")
    .Input("wall_value: float32")
    .Input("free_value: float32")
    .Input("wall_value_not_visible: float32")
    .Input("not_reachable: float32")
    .Input("free_not_visible_lower: float32")
    .Input("free_not_visible_upper: float32")
    .Input("wall_visible_factor: float32")
    .Input("wall_non_floor_or_wall_factor: float32")
    .Input("true_wall_factor: float32")
    .Output("loss: float32");

#include "tensorflow/core/framework/op_kernel.h"
#include "LossValueType.h"

using namespace tensorflow;

class LossConversionOp : public OpKernel {
 public:
  explicit LossConversionOp(OpKernelConstruction* context) : OpKernel(context) {}

  float extractInputScalar(OpKernelContext* context, const int id, const std::string& name){
    const Tensor& tensor = context->input(id);
    return tensor.scalar<float>().data()[0];
  }

  void Compute(OpKernelContext* context) override {
    // Grab the input tensor
    const Tensor& input_tensor = context->input(0);
    auto input = input_tensor.tensor<uint8, 3>();

    const float wallValue = extractInputScalar(context, 1, "wall_value");
    const float freeValue = extractInputScalar(context, 2, "free_value");
    const float wallValueNotVisible  = extractInputScalar(context, 3, "wall_value_not_visible");
    const float notReachable = extractInputScalar(context, 4, "not_reachable");
    const float freeNotVisibleLower = extractInputScalar(context, 5, "free_not_visible_lower");
    const float freeNotVisibleUpper = extractInputScalar(context, 6, "free_not_visible_upper");
    const float factorVisible = extractInputScalar(context, 7, "wall_visible_factor");
    const float factorNonFloorOrWall = extractInputScalar(context, 8, "wall_non_floor_or_wall_factor");
    const float factorTrueWall = extractInputScalar(context, 9, "true_wall_factor");

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
                            float lossValue = 0.0;
                            const unsigned char currentValue = input(x * blockSize + i, y * blockSize + j, z * blockSize + k);
                            if(currentValue == LossValueType::wallValue() || currentValue == LossValueType::trueWallValue()){
                                lossValue = wallValue;
                            }else if (currentValue == LossValueType::freeValue()){
                                lossValue = freeValue;
                            }else if (currentValue == LossValueType::notReachable()){
                                lossValue = notReachable;
                            }else if (currentValue == LossValueType::wallValueNotVisible() || currentValue == LossValueType::trueWallValueNotVisible()){
                                lossValue = wallValueNotVisible;
                            }else if (currentValue >= LossValueType::freeNotVisibleStart() && currentValue <= LossValueType::freeNotVisibleEnd()){
                                const float factor = float (currentValue - LossValueType::freeNotVisibleStart()) / float (LossValueType::freeNotVisibleEnd() - LossValueType::freeNotVisibleStart());
                                lossValue = factor * (freeNotVisibleUpper - freeNotVisibleLower) + freeNotVisibleLower;
                            }else if(currentValue >= lowerClassEndNumber && currentValue < upperClassEndNumber){
                                // this value is now from 0 to 4 * amountOfClasses
                                const int classNr = (currentValue - lowerClassEndNumber) % amountOfClasses;
                                lossValue = wallValue;
                                if(classNr == 2 || classNr == 8){
                                    // 2 is wall and 8 is floor -> both are easy and should only get a medium high loss
                                    // lossValue *= 1; // doesn't have to be performed as long as factor is one
                                }else{
                                    // is one of the following void, table, bath, sofa, cabinet, bed, chair, lighting
                                    lossValue *= factorNonFloorOrWall;
                                }
                                if(currentValue >= LossValueType::getClassValueTrueWall(0) && currentValue < LossValueType::getClassValueTrueWall(amountOfClasses)){
                                    // visible value and also true value
                                    lossValue *= factorVisible * factorTrueWall;
                                }else if(currentValue >= LossValueType::getClassValue(0, amountOfClasses) && currentValue < LossValueType::getClassValue(amountOfClasses, amountOfClasses)){
                                    // visible value but not true value
                                    lossValue *= factorVisible;
                                }else if(currentValue >= LossValueType::getClassValueNotVisibleTrueWall(0, amountOfClasses) && currentValue < LossValueType::getClassValueNotVisibleTrueWall(amountOfClasses, amountOfClasses)){
                                    // not visible value and true value
                                    lossValue *= factorTrueWall;
                                }// else is only not visible and not true value -> no factor there
                            }
                            mean_value = fac * float(lossValue) + (1.0 - fac) * mean_value;
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


REGISTER_KERNEL_BUILDER(Name("LossConversion").Device(DEVICE_CPU), LossConversionOp);