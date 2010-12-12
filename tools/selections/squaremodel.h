#ifndef SQUAREMODEL_H
#define SQUAREMODEL_H

#include "tfr/freqaxis.h"
#include "signal/operation.h"
#include "heightmap/position.h"

namespace Sawe {
    class Project;
}

namespace Tools { namespace Selections
{
    class SquareModel
    {
    public:
        enum SquareType
        {
            SquareType_SquareSelection,
            SquareType_FrequencySelection,
            SquareType_TimeSelection
        };

        SquareModel( Tfr::FreqAxis const& fa, Sawe::Project* p );
        ~SquareModel();

        void updateFilter();
        Signal::pOperation filter;

        Heightmap::Position a, b;
        SquareType type;

        void validate();
    private:
        void createFilter();
        Tfr::FreqAxis fa_;
        Sawe::Project* project_;
    };
} } // namespace Tools::Selections
#endif // SQUAREMODEL_H

