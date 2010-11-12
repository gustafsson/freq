#ifndef PEAKMODEL_H
#define PEAKMODEL_H

#include "support/peakfilter.h"
#include "signal/operation.h"
#include "heightmap/reference.h"

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
    PeakModel();

    /**
      Get the PeakFilter
      */
    Filters::PeakFilter* peak_filter();

    Signal::pOperation filter;

    void findAddPeak( Heightmap::Reference ref, Heightmap::Position pos );

private:
    typedef boost::shared_ptr< GpuCpuData<bool> > PeakAreaP;
    typedef boost::unordered_map<Heightmap::Reference, PeakAreaP> PeakAreas;
    PeakAreas classifictions;
    //PeakAreas gaussed_classifictions;

    PeakAreaP getPeakArea(Heightmap::Reference);
    //PeakAreaP getPeakAreaGauss(Heightmap::Reference);
    bool classifiedVal(unsigned x, unsigned y, unsigned w, unsigned h);
    /*float& gaussedVal(unsigned x, unsigned y, unsigned w, unsigned h);
    void smearGauss();*/
    void recursivelyClassify( Heightmap::Reference ref,
                              unsigned w, unsigned h,
                              unsigned x, unsigned y,
                              PropagationState prevState, float prevVal );
    void recursivelyClassify( Heightmap::Reference ref,
                              float *data, float* classification,
                              unsigned w, unsigned h,
                              unsigned x, unsigned y,
                              PropagationState prevState, float prevVal );
};

}} // Tools::Selections

#endif // PEAKMODEL_H
