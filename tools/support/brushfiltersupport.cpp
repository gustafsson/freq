#include "brushfiltersupport.h"
#include "brushfilter.h"
#include "sawe/application.h"

namespace Tools {
namespace Support {

BrushFilterSupport::
        BrushFilterSupport(BrushFilter*parent) :
    bf_(parent)
{
    connect(Sawe::Application::global_ptr(), SIGNAL(clearCachesSignal()), SLOT(release_resources()));
}

void BrushFilterSupport::
        release_resources()
{
    bf_->release_extra_resources();
}

} // namespace Support
} // namespace Tools
