#ifndef RECTANGLEMODEL_H
#define RECTANGLEMODEL_H

#include "tfr/freqaxis.h"
#include "signal/operation.h"
#include "heightmap/position.h"

namespace Sawe {
    class Project;
}

namespace Tools { namespace Selections
{
    class RectangleModel
    {
    public:
        enum RectangleType
        {
            RectangleType_RectangleSelection,
            RectangleType_FrequencySelection,
            RectangleType_TimeSelection
        };

        RectangleModel( Tfr::FreqAxis const& fa, Sawe::Project* p );
        ~RectangleModel();

        Signal::pOperation updateFilter();
        void tryFilter(Signal::pOperation o);

        Heightmap::Position a, b;
        RectangleType type;

        void validate();
    private:
        void createFilter();
        Tfr::FreqAxis fa_;
        Sawe::Project* project_;
    };
} } // namespace Tools::Selections
#endif // RECTANGLEMODEL_H

