#pragma once

/*
 A continous brush stroke results in an image. This image is then applied upon
 the data through a filter which interprets what to do with the image data.
 */

#include "tfr/filter.h"
#include "sawe/project.h"
#include "support/brushfilter.h"

namespace Tools
{

class BrushModel
{
public:
    BrushModel( Sawe::pProject project );

    /**
      Get the BrushFilter.
      */
    Support::BrushFilter* filter();

private:
    Signal::pOperation filter_;
};


} // namespace Tools
