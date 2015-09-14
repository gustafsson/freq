#ifndef FANTRACKERVIEW_H
#define FANTRACKERVIEW_H

#include <QObject>
#include "tools/renderview.h"
#include "support/fantrackerfilter.h"
#include "fantrackermodel.h"

namespace Tools {

class FanTrackerView : public QObject
{
    Q_OBJECT
public:
    explicit FanTrackerView(FanTrackerModel*, RenderView*);

    FanTrackerModel* model_;

private:
    RenderView* render_view_;

signals:

public slots:
    virtual void draw();

};

} // namespace Tools

#endif // FANTRACKERVIEW_H
