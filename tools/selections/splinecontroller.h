#ifndef SPLINECONTROLLER_H
#define SPLINECONTROLLER_H

#include "splineview.h"
#include "tools/selectioncontroller.h"
#include "heightmap/position.h"

#include <QWidget>

namespace Tools { namespace Selections
{

class SplineController: public QWidget
{
    Q_OBJECT
public:
    SplineController(
            SplineView* view,
            SelectionController* selection_controller);
    ~SplineController();

signals:
    void enabledChanged(bool active);

private slots:
    void enableSplineSelection(bool active);

private:
    // Event handlers
    virtual void mousePressEvent ( QMouseEvent * e );
    virtual void mouseReleaseEvent ( QMouseEvent * e );
    virtual void mouseMoveEvent ( QMouseEvent * e );
    virtual void changeEvent ( QEvent * event );

    // View
    SplineView* view_;
    SplineModel* model() { return view_->model_; }

    // GUI
    void setupGui();
    Qt::MouseButton selection_button_;
    Qt::MouseButton stop_button_;
    Tools::SelectionController* selection_controller_;

    // State
};

}} // namespace Tools::Selections

#endif // SPLINECONTROLLER_H
