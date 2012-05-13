#ifndef PEAKMODEL_H
#define PEAKMODEL_H

#include "support/splinefilter.h"
#include "signal/operation.h"
#include "heightmap/reference.h"
#include <boost/unordered_map.hpp>
#include "splinemodel.h"

namespace Tools { namespace Selections
{

enum PropagationState {
    PS_Increasing,
    PS_Decreasing,
    PS_Out
};

class PeakModel
{
public:
    PeakModel( RenderModel* rendermodel );

    Signal::pOperation updateFilter() { return spline_model.updateFilter(); }

    SplineModel spline_model;

    void findAddPeak( Heightmap::Reference ref, Heightmap::Position pos );

private:
    struct BorderCoordinates
    {
        BorderCoordinates(unsigned x=0, unsigned y=0):x(x), y(y) {}
        unsigned x, y;
    };

    typedef DataStorage<bool>::Ptr PeakAreaP;
    typedef boost::unordered_map<Heightmap::Reference, PeakAreaP> PeakAreas;

    PeakAreas classifictions;

    void findBorder();
    std::vector<BorderCoordinates> border_nodes;
    unsigned pixel_count;

    float found_max;
    float found_min;
    float middle_limit;
    float min_limit;
    unsigned pixel_limit;
    bool use_min_limit;

    bool anyBorderPixel( BorderCoordinates&, unsigned w, unsigned h );
    BorderCoordinates nextBorderPixel( BorderCoordinates, unsigned w, unsigned h, unsigned& firstdir );

    PeakAreaP getPeakArea(Heightmap::Reference);
    bool classifiedVal(unsigned x, unsigned y, unsigned w, unsigned h);
    float heightVal(Heightmap::Reference ref, unsigned x, unsigned y);
    void recursivelyClassify( Heightmap::Reference ref,
                              unsigned w, unsigned h,
                              unsigned x, unsigned y,
                              PropagationState prevState, float prevVal );
    void recursivelyClassify( Heightmap::Reference ref,
                              float *data, bool* classification,
                              unsigned w, unsigned h,
                              unsigned x, unsigned y,
                              PropagationState prevState, float prevVal );
    void loopClassify( Heightmap::Reference ref,
                       unsigned x, unsigned y );

    //PeakAreas gaussed_classifictions;
    //PeakAreaP getPeakAreaGauss(Heightmap::Reference);
    /*float& gaussedVal(unsigned x, unsigned y, unsigned w, unsigned h);
    void smearGauss();*/
};

}} // Tools::Selections

#endif // PEAKMODEL_H
