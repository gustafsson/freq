#ifndef ELLIPSEMODEL_H
#define ELLIPSEMODEL_H

#include "tfr/freqaxis.h"
#include "signal/operation.h"
#include "heightmap/position.h"


namespace Tools { namespace Selections
{
    class EllipseModel
    {
    public:
        EllipseModel( Tfr::FreqAxis const& fa );
        ~EllipseModel();

        void updateFilter();
        Signal::pOperation filter;

        Heightmap::Position a, b;

    private:
        void createFilter();
        Tfr::FreqAxis fa_;
    };
} } // namespace Tools::Selections
#endif // ELLIPSEMODEL_H

