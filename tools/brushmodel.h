#pragma once

/*
 A continous brush stroke results in an image. This image is then applied upon
 the data through a filter which interprets what to do with the image data.
 */

#include "tfr/filter.h"
#include "sawe/project.h"
#include "support/brushfilter.h"
#include "heightmap/reference.h"

namespace Tools
{

class BrushModel
{
public:
    BrushModel( Sawe::Project* project );

    /**
      Get the BrushFilter.
      */
    Support::BrushFilter* filter();

    float brush_factor;

    Signal::Interval paint( Heightmap::Reference ref, Heightmap::Position pos );

private:
    Signal::pOperation filter_;

    Signal::Interval addGauss( Heightmap::Reference ref, Heightmap::Position pos );
};


} // namespace Tools
