#ifndef MATLABOPERATIONWIDGET_H
#define MATLABOPERATIONWIDGET_H

#include "signal/operation.h"
#include "signal/target.h"

#include "adapters/matlaboperation.h"

#include <QWidget>

namespace Sawe { class Project; }

namespace Tools {

namespace Ui {
    class MatlabOperationWidget;
}

class MatlabOperationWidget : public QWidget, public Adapters::MatlabOperationSettings
{
    Q_OBJECT

public:
    explicit MatlabOperationWidget(Sawe::Project* project, QWidget *parent = 0);
    ~MatlabOperationWidget();

    std::string scriptname();
    void scriptname(std::string);

    virtual int chunksize();
    void chunksize(int);

    virtual bool computeInOrder();
    void computeInOrder(bool);

    virtual int redundant();
    void redundant(int);

    Adapters::MatlabOperation* operation;

private slots:
    void browse();

    void populateTodoList();

private:
    Ui::MatlabOperationWidget *ui;

    Signal::pTarget target;
    Sawe::Project* project;
};


} // namespace Tools
#endif // MATLABOPERATIONWIDGET_H
