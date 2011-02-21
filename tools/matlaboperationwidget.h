#ifndef MATLABOPERATIONWIDGET_H
#define MATLABOPERATIONWIDGET_H

#include "signal/operation.h"

#include <QWidget>

namespace Tools {

namespace Ui {
    class MatlabOperationWidget;
}

class MatlabOperationWidget : public QWidget
{
    Q_OBJECT

public:
    explicit MatlabOperationWidget(unsigned FS, QWidget *parent = 0);
    ~MatlabOperationWidget();

    Signal::pOperation createMatlabOperation();

    std::string scriptname();
    void scriptname(std::string);

    int chunksize();
    void chunksize(int);

    bool computeInOrder();
    void computeInOrder(bool);

    int redundant();
    void redundant(int);

private slots:
    void browse();

private:
    Ui::MatlabOperationWidget *ui;
};


} // namespace Tools
#endif // MATLABOPERATIONWIDGET_H
