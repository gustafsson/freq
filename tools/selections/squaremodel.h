#ifndef SQUAREMODEL_H
#define SQUAREMODEL_H

#include "tfr/freqaxis.h"
#include "signal/operation.h"
#include "heightmap/position.h"


namespace Tools { namespace Selections
{
    class SquareModel
    {
    public:
        SquareModel( Tfr::FreqAxis const& fa );
        ~SquareModel();

        void updateFilter();
        Signal::pOperation filter;

        Heightmap::Position a, b;

    private:
        void createFilter();
        Tfr::FreqAxis fa_;
    };
} } // namespace Tools::Selections
#endif // SQUAREMODEL_H

