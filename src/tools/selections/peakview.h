#ifndef PEAKVIEW_H
#define PEAKVIEW_H

#include <QObject>
#include "splineview.h"

namespace Tools { namespace Selections
{

class PeakModel;

class PeakView: public QObject
{
    Q_OBJECT
public:
    PeakView(PeakModel* model);
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
};

}} // namespace Tools::Selections

#endif // PEAKVIEW_H
