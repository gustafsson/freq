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
        ~TimelineController();

    private:
        RenderModel *model;
        TimelineView *view;

    private slots:
        void hideTimeline();

    private:
        // GUI
        // These are never used outside setupGui, but they are named here
        // to make it clear what class that is "responsible" for them.
        // By responsible I mean to create them, insert them to their proper
        // place in the GUI and take care of events. The objects lifetime
        // depends on the parent QObject which they are inserted into.
        QDockWidget* dock;

        void setupGui();

        // UI events
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
