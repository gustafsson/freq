#ifndef TIMELINECONTROLLER_H
#define TIMELINECONTROLLER_H

#include "ui/mousecontrol.h"

#include <QWidget>

class QDockWidget;

namespace Tools
{
    class RenderModel;
    class TimelineView;

    class TimelineController: public QWidget
    {
        Q_OBJECT
    public:
        TimelineController( TimelineView* timeline_view );

    private:
        RenderModel *model;
        TimelineView *view;

    private: // GUI
        // These are never used outside setupGui, but they are named here
        // to make it clear what class that is responsible for them.
        QDockWidget* dock;

        void setupGui();

    private: // UI events
        // overloaded from QWidget
        virtual void wheelEvent ( QWheelEvent *e );
        virtual void mousePressEvent ( QMouseEvent * e );
        virtual void mouseReleaseEvent ( QMouseEvent * e);
        virtual void mouseMoveEvent ( QMouseEvent * e );

        Ui::MouseControl moveButton;
        int _movingTimeline;
    };
}

#endif // TIMELINECONTROLLER_H
