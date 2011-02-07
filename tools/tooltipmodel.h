#ifndef TOOLTIPMODEL_H
#define TOOLTIPMODEL_H

// Tools
#include "commentview.h"

// Sonic AWE
#include "heightmap/reference.h"

// Qt
#include <QPointer>

namespace Tools {
class CommentController;

class TooltipModel
{
public:
    TooltipModel(RenderView *render_view, CommentController* comments );

    const Heightmap::Position& comment_pos();

    void showToolTip( Heightmap::Position p );
    unsigned guessHarmonicNumber( const Heightmap::Position& pos );
    float computeMarkerMeasure(const Heightmap::Position& pos, unsigned i, Heightmap::Reference* ref=0);

    Heightmap::Position pos;
    float frequency;
    float max_so_far;
    unsigned markers;
    QPointer<CommentView> comment;
    enum AutoMarkersState
    {
        ManualMarkers,
        AutoMarkerWorking,
        AutoMarkerFinished
    } automarking;

private:
    CommentController* _comments;
    RenderView *render_view_;
    unsigned fetched_heightmap_values;
};

} // namespace Tools

#endif // TOOLTIPMODEL_H
