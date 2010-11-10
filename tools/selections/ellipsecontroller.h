#ifndef ELLIPSECONTROLLER_H
#define ELLIPSECONTROLLER_H

#include "ellipseview.h"
#include "tools/selectioncontroller.h"
#include "heightmap/position.h"

#include <QWidget>

namespace Tools { namespace Selections
{

class EllipseController: public QWidget
{
    Q_OBJECT
public:
    EllipseController(
            EllipseView* view,
            SelectionController* selection_controller);
    ~EllipseController();

signals:
    void enabledChanged(bool active);

private slots:
    void enableEllipseSelection(bool active);
/*    void enableSquareSelection(bool active);
    void enableSplineSelection(bool active);
    void enablePolygonSelection(bool active);
    void enablePeakSelection(bool active);*/

private:
    // Event handlers
    virtual void mousePressEvent ( QMouseEvent * e );
    virtual void mouseReleaseEvent ( QMouseEvent * e );
    virtual void mouseMoveEvent ( QMouseEvent * e );
    virtual void changeEvent ( QEvent * event );

    // View
    EllipseView* view_;
    EllipseModel* model() { return view_->model_; }

    // GUI
    void setupGui();
    Qt::MouseButton selection_button_;
    Tools::SelectionController* selection_controller_;

    // State
    Heightmap::Position selectionStart;
};

}} // namespace Selections::Tools

#endif // ELLIPSECONTROLLER_H
