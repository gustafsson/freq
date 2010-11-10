#ifndef SQUARECONTROLLER_H
#define SQUARECONTROLLER_H

#include "squareview.h"
#include "tools/selectioncontroller.h"
#include "heightmap/position.h"

#include <QWidget>

namespace Tools { namespace Selections
{

class SquareController: public QWidget
{
    Q_OBJECT
public:
    SquareController(
            SquareView* view,
            SelectionController* selection_controller);
    ~SquareController();

signals:
    void enabledChanged(bool active);

private slots:
    void enableSquareSelection(bool active);

private:
    // Event handlers
    virtual void mousePressEvent ( QMouseEvent * e );
    virtual void mouseReleaseEvent ( QMouseEvent * e );
    virtual void mouseMoveEvent ( QMouseEvent * e );
    virtual void changeEvent ( QEvent * event );

    // View
    SquareView* view_;
    SquareModel* model() { return view_->model_; }

    // GUI
    void setupGui();
    Qt::MouseButton selection_button_;
    Tools::SelectionController* selection_controller_;

    // State
    Heightmap::Position selectionStart;
};

}} // namespace Selections::Tools

#endif // SQUARECONTROLLER_H
