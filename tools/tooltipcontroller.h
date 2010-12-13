#ifndef TOOLTIPCONTROLLER_H
#define TOOLTIPCONTROLLER_H

#include "ui/mousecontrol.h"
#include "heightmap/position.h"
#include "commentcontroller.h"

#include <QWidget>

namespace Tools
{
    class RenderView;
    class RenderModel;

    class TooltipController: public QWidget
    {
        Q_OBJECT
    public:
        TooltipController(RenderView *view, RenderModel *model, CommentController* comments);
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
        virtual void changeEvent(QEvent *);
        void showToolTip(Heightmap::Position p);

        // Model and View
        RenderView* _view;
        RenderModel* _model;
        CommentController* _comments;

        // GUI
        void setupGui();

        // State
        Ui::MouseControl infoToolButton;
        float _prev_f;
        float _current_f;
        float _max_so_far;
        CommentView* comment;
    };
}

#endif // TOOLTIPCONTROLLER_H
