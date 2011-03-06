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

private:
    WorkerView* view_;
};

} // namespace Tools

#endif // WORKERCONTROLLER_H
