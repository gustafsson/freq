#ifndef RECTANGLECONTROLLER_H
#define RECTANGLECONTROLLER_H

#include "rectangleview.h"
#include "tools/selectioncontroller.h"
#include "heightmap/position.h"

#include <QWidget>

namespace Tools { namespace Selections
{

class RectangleForm;

class RectangleController: public QWidget
{
    Q_OBJECT
public:
    RectangleController(
            RectangleView* view,
            SelectionController* selection_controller);
    ~RectangleController();

signals:
    void enabledChanged(bool active);

private slots:
    void enableRectangleSelection(bool active);
    void enableTimeSelection(bool active);
    void enableFrequencySelection(bool active);
    void selectionChanged();

private:
    // Event handlers
    virtual void mousePressEvent ( QMouseEvent * e );
    virtual void mouseReleaseEvent ( QMouseEvent * e );
    virtual void mouseMoveEvent ( QMouseEvent * e );
    virtual void changeEvent ( QEvent * event );

    void enableSelectionType(const RectangleModel::RectangleType type, const bool active);

    // View
    RectangleView* view_;
    QPointer<RectangleForm> rectangleForm_;
    RectangleModel* model() { return view_->model_; }

    // GUI
    void setupGui();
    Qt::MouseButton selection_button_;
    Tools::SelectionController* selection_controller_;

    // State
    Heightmap::Position selectionStart;
    QScopedPointer<Ui::ComboBoxAction> one_action_at_a_time_;
};

}} // namespace Tools::Selections

#endif // RECTANGLECONTROLLER_H
