#if 0
#ifndef RECTANGLEMODEL_H
#define RECTANGLEMODEL_H

#include "tfr/freqaxis.h"
#include "signal/operation.h"
#include "heightmap/position.h"

namespace Sawe {
    class Project;
}

namespace Tools { class RenderModel; }

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

        RectangleModel( RenderModel* rendermodel, Sawe::Project* p );
        ~RectangleModel();

        Signal::pOperation updateFilter();
        bool tryFilter(Signal::pOperation o);
        bool replaceFilter( Signal::pOperation filter );

        Heightmap::Position a, b;
        RectangleType type;

        void validate();
        Tfr::FreqAxis freqAxis();
        Sawe::Project* project() { return project_; }

    private:
        void createFilter();
        RenderModel* rendermodel_;
        Sawe::Project* project_;
    };
} } // namespace Tools::Selections
#endif // RECTANGLEMODEL_H
#endif
