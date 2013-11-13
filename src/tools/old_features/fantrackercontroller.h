#if 0
#ifndef FANTRACKERCONTROLLER_H
#define FANTRACKERCONTROLLER_H

#include <QObject>
#include "renderview.h"
#include "fantrackerview.h"
#include "tools/support/fantrackerfilter.h"

namespace Tools {

class FanTrackerController : public QObject
{
    Q_OBJECT
public:
    explicit FanTrackerController(FanTrackerView*, RenderView*);

signals:

public slots:

private slots:
    void receiveToggleFanTracker(bool);

private:
    void setupGui();
    RenderView* render_view_;
    FanTrackerView* view_;
};

} // namespace Tools

#endif // FANTRACKERCONTROLLER_H
#endif
