#ifndef SPLINEMODEL_H
#define SPLINEMODEL_H

#include "tfr/freqaxis.h"
#include "signal/operation.h"
#include "heightmap/position.h"

#include <vector>

namespace Tools { class RenderModel; }

namespace Tools { namespace Selections
{
    class SplineModel
    {
    public:
        SplineModel( RenderModel* rendermodel );
        ~SplineModel();

        Signal::OperationDesc::Ptr updateFilter();

        std::vector<Heightmap::Position> v;
        bool drawing;
        Tfr::FreqAxis freqAxis();

    private:
        void createFilter();
        RenderModel* rendermodel_;
    };
} } // namespace Tools::Selections
#endif // SPLINEMODEL_H

