#ifndef PEAKVIEW_H
#define PEAKVIEW_H

#include <QObject>
#include "signal/worker.h"
#include "splineview.h"

namespace Tools { namespace Selections
{

class PeakModel;

class PeakView: public QObject
{
    Q_OBJECT
public:
    PeakView(PeakModel* model, Signal::Worker* worker);
    ~PeakView();

    void drawSelectionPeak();

    bool visible, enabled;

public slots:
    /// Connected in PeakController
    virtual void draw();

private:
    SplineView spline_view;

    friend class PeakController;
    PeakModel* model_;
    Signal::Worker* worker_;
};

}} // namespace Tools::Selections

#endif // PEAKVIEW_H
