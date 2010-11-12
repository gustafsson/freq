#ifndef PEAKVIEW_H
#define PEAKVIEW_H

#include <QObject>
#include "signal/worker.h"

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

    bool enabled;

public slots:
    /// Connected in SquareController
    virtual void draw();

private:
    friend class PeakController;
    PeakModel* model_;
    Signal::Worker* worker_;
};

}} // namespace Selections::Tools

#endif // PEAKVIEW_H
