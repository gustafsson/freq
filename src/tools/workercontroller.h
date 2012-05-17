#ifndef WORKERCONTROLLER_H
#define WORKERCONTROLLER_H

#include "workerview.h"

namespace Tools {

class WorkerController : public QObject
{
    Q_OBJECT
public:
    WorkerController(WorkerView* view, class RenderView* renderview, class TimelineView* timelineview);

signals:

public slots:
    void setEnabled( bool enabled );

private:
    WorkerView* view_;
    class RenderView* renderview_;
    class TimelineView* timelineview_;
};

} // namespace Tools

#endif // WORKERCONTROLLER_H
