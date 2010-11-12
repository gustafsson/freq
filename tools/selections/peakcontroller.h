#ifndef PEAKCONTROLLER_H
#define PEAKCONTROLLER_H

#include "peakview.h"
#include "tools/selectioncontroller.h"
#include "heightmap/position.h"

#include <QWidget>

namespace Tools { namespace Selections
{

class PeakController: public QWidget
{
    Q_OBJECT
public:
    PeakController(
            PeakView* view,
            SelectionController* selection_controller);
    ~PeakController();

signals:
    void enabledChanged(bool active);

private slots:
    void enablePeakSelection(bool active);

private:
    // Event handlers
    virtual void mousePressEvent ( QMouseEvent * e );
    virtual void mouseReleaseEvent ( QMouseEvent * e );
    virtual void mouseMoveEvent ( QMouseEvent * e );
    virtual void changeEvent ( QEvent * event );

    // View
    PeakView* view_;
    PeakModel* model() { return view_->model_; }

    // GUI
    void setupGui();
    Qt::MouseButton selection_button_;
    Tools::SelectionController* selection_controller_;
};

}} // namespace Selections::Tools

#endif // PEAKCONTROLLER_H
