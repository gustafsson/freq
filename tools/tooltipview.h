#ifndef TOOLTIPVIEW_H
#define TOOLTIPVIEW_H

#include <QObject>
#include "signal/worker.h"
#include "tooltipmodel.h"
#include "renderview.h"

namespace Tools {

class TooltipView : public QObject
{
    Q_OBJECT
public:
    TooltipView(TooltipModel* model, RenderView* render_view);
    ~TooltipView();

    void drawMarkers();
    void drawMarker( Heightmap::Position p );

    bool enabled;
    bool visible;

public slots:
    /// Connected in SquareController
    virtual void draw();

private:
    friend class TooltipController;
    TooltipModel* model_;
    RenderView* render_view_;
};

} // namespace Tools

#endif // TOOLTIPVIEW_H
