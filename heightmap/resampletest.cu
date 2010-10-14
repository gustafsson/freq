#include "resampletest.cu.h"
#include "resample.cu.h"

void simple_resample2d(
        cudaPitchedPtrType<float2> input,
        cudaPitchedPtrType<float2> output
        )
{
    resample2d_plain<float2, float2,NoConverter<float2, float2>, NoTranslation>(
            input,
            output
    );
}
