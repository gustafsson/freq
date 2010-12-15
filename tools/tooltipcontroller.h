#ifndef TOOLTIPCONTROLLER_H
#define TOOLTIPCONTROLLER_H

#include "ui/mousecontrol.h"
#include "heightmap/position.h"
#include "commentcontroller.h"

#include "tooltipview.h"

#include <QWidget>

namespace Tools
{
    class RenderView;
    class RenderModel;

    class TooltipController: public QWidget
    {
        Q_OBJECT
    public:
        TooltipController(TooltipView* view,
                          RenderView *render_view,
                          CommentController* comments);
        ~TooltipController();

    signals:
        void enabledChanged(bool active);

    private slots:
        virtual void receiveToggleInfoTool(bool);

    private:
        // Event handlers
        virtual void mousePressEvent ( QMouseEvent * e );
        virtual void mouseReleaseEvent ( QMouseEvent * e );
        virtual void mouseMoveEvent ( QMouseEvent * e );
        virtual void wheelEvent(QWheelEvent *);
        virtual void changeEvent(QEvent *);
        void showToolTip( Heightmap::Position p );
        unsigned guessHarmonicNumber( const Heightmap::Position& pos );
        float computeMarkerMeasure(const Heightmap::Position& pos, unsigned i, Heightmap::Reference* ref=0);


        // Model and View
        TooltipView* view_;
        TooltipModel* model() { return view_->model_; }
        RenderView* render_view_;
        CommentController* _comments;

        // GUI
        void setupGui();

        // State
        Ui::MouseControl infoToolButton;
        unsigned fetched_heightmap_values;
    };
}

#endif // TOOLTIPCONTROLLER_H
