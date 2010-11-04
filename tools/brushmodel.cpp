#include "brushmodel.h"
#include <demangle.h>

namespace Tools {


BrushModel::
        BrushModel( Sawe::pProject project )
{
    filter_.reset( new Support::MultiplyBrush );
    filter_->source( project->head_source() );
    project->head_source( filter_ );
}


Support::BrushFilter* BrushModel::
        filter()
{
    return dynamic_cast<Support::BrushFilter*>(filter_.get());
}


} // namespace Tools
