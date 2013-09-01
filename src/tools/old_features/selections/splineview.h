#if 0

#ifndef SPLINEVIEW_H
#define SPLINEVIEW_H

#include <QObject>

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

    bool visible, enabled;

public slots:
    /// Connected in SplineController
    virtual void draw();

private:
    friend class SplineController;
    SplineModel* model_;
    Signal::Worker* worker_;
};

}} // namespace Tools::Selections

#endif // SPLINEVIEW_H
#endif
