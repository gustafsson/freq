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

        const std::list<QPointer<TooltipView> >& views() const { return views_; }
        void setCurrentView(TooltipView* value );
        TooltipView* current_view();

    signals:
        void enabledChanged(bool active);
        void tooltipChanged();

    private slots:
        void receiveToggleInfoTool(bool);
        void emitTooltipChanged();

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
        QPointer<TooltipView> current_view_;

        // GUI
        void setupGui();

        void setupView(TooltipView* view);

        // State
        Ui::MouseControl infoToolButton;
    };
}

#endif // TOOLTIPCONTROLLER_H
