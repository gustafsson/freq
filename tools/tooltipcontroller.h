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
    class TooltipView;
    class TooltipModel;

    class TooltipController: public QWidget
    {
        Q_OBJECT
    public:
        TooltipController(RenderView *render_view,
                          CommentController* comments);
        ~TooltipController();

        void setCurrentView(TooltipView* value ) { current_view_ = value; }

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


        // Model and View
        std::list<QPointer<TooltipView> > views_;
        RenderView* render_view_;
        CommentController* comments_;

        TooltipModel* current_model();
        TooltipView* current_view_;

        // GUI
        void setupGui();

        void setupView(TooltipView* view);

        // State
        Ui::MouseControl infoToolButton;
    };
}

#endif // TOOLTIPCONTROLLER_H
