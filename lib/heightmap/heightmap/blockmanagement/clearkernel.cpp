#include "clearkernel.h"
#include "operate.h"

class SetZero
{
public:
    SetZero(float limit):limit(limit) {}

    template<typename T>
    RESAMPLE_CALL void operator()(T& e, ResamplePos const& v)
    {
        if (v.x >= limit)
            e = 0;
    }
private:
    float limit;
};

extern "C"
void blockClearPart( BlockData::ptr block,
                 int start_t )
{
    element_operate(block, ResampleArea(0,0, block->size().width, 1), SetZero(start_t));
}
