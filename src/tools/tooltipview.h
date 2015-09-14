#ifndef TOOLTIPVIEW_H
#define TOOLTIPVIEW_H

#include <QObject>
#include "tooltipmodel.h"
#include "tools/renderview.h"

namespace Tools {

class TooltipController;
class ToolRepo;

class TooltipView : public QObject
{
    Q_OBJECT
public:
    TooltipView(TooltipModel* model,
                TooltipController* controller,
                RenderView* render_view);
    ~TooltipView();

    void drawMarkers();
    void drawMarker( Heightmap::Position p );
    TooltipModel* model() { return model_; }

    bool enabled;
    bool visible;

signals:
    void tooltipChanged();

public slots:
    /// Connected in constructor
    void draw();

    void setHidden(bool value);
    void setFocus();
    void seppuku();

private:
    bool initialized;
    void initialize();

    TooltipModel* model_;
    TooltipController* controller_;
    RenderView* render_view_;
    Heightmap::Position prev_pos_;
};

} // namespace Tools

#endif // TOOLTIPVIEW_H
