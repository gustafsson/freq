#ifndef SPLINEVIEW_H
#define SPLINEVIEW_H

#include <QObject>
#include "signal/worker.h"

namespace Tools { namespace Selections
{

class SplineModel;

class SplineView: public QObject
{
    Q_OBJECT
public:
    SplineView(SplineModel* model, Signal::Worker* worker);
    ~SplineView();

    void drawSelectionSpline();

    bool enabled;

public slots:
    /// Connected in SquareController
    virtual void draw();

private:
    friend class SplineController;
    SplineModel* model_;
    Signal::Worker* worker_;
};

}} // namespace Selections::Tools

#endif // SPLINEVIEW_H
