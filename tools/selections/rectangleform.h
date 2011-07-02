#ifndef RECTANGLEFORM_H
#define RECTANGLEFORM_H

#include <QWidget>

namespace Tools {
namespace Selections {

namespace Ui {
    class RectangleForm;
}

class RectangleModel;

class RectangleForm : public QWidget
{
    Q_OBJECT

public:
    explicit RectangleForm(RectangleModel* model, QWidget *parent = 0);
    ~RectangleForm();

    void updateGui();
    void showAsCurrentTool( bool isCurrent );

private slots:
    void updateSelection();

private:
    Ui::RectangleForm *ui;
    RectangleModel* model_;
    bool dontupdate_;
};


} // namespace Selections
} // namespace Tools
#endif // RECTANGLEFORM_H
