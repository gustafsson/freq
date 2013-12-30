#ifndef ELLIPSEMODEL_H
#define ELLIPSEMODEL_H

#include "tfr/freqaxis.h"
#include "signal/operation.h"
#include "heightmap/position.h"

namespace Tools { class RenderModel; }

namespace Tools { namespace Selections
{
    class EllipseModel
    {
    public:
        EllipseModel( RenderModel* rendermodel );
        ~EllipseModel();

        Signal::OperationDesc::Ptr updateFilter();
        void tryFilter(Signal::OperationDesc::Ptr o);

        Heightmap::Position centre, centrePlusRadius;
        Tfr::FreqAxis freqAxis();

    private:
        void createFilter();
        RenderModel* rendermodel_;
    };
} } // namespace Tools::Selections
#endif // ELLIPSEMODEL_H

