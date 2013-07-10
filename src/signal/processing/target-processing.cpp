#include "target.h"

namespace Signal {
namespace Processing {

Target::
        Target(Step::Ptr step)
    :
      step_(step)
{
}


Step::Ptr Target::
        step()
{
    return step_;
}


} // namespace Processing
} // namespace Signal
