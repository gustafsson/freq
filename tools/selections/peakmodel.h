#ifndef PEAKMODEL_H
#define PEAKMODEL_H

#include "support/peakfilter.h"
#include "signal/operation.h"
#include "heightmap/reference.h"

namespace Tools { namespace Selections
{

class PeakModel
{
public:
    PeakModel();

    /**
      Get the PeakFilter
      */
    Filters::PeakFilter* peak_filter();

    Signal::pOperation filter;

    void findAddPeak( Heightmap::Reference ref, Heightmap::Position pos );
};

}} // Tools::Selections

#endif // PEAKMODEL_H
