#ifndef SPLINEMODEL_H
#define SPLINEMODEL_H

#include "tfr/freqaxis.h"
#include "signal/operation.h"
#include "heightmap/position.h"


namespace Tools { namespace Selections
{
    class SplineModel
    {
    public:
        SplineModel( Tfr::FreqAxis const& fa );
        ~SplineModel();

        void updateFilter();
        Signal::pOperation filter;

        std::vector<Heightmap::Position> v;
        bool drawing;

    private:
        void createFilter();
        Tfr::FreqAxis fa_;
    };
} } // namespace Tools::Selections
#endif // SPLINEMODEL_H

