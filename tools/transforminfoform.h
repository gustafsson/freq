#ifndef TRANSFORMINFOFORM_H
#define TRANSFORMINFOFORM_H

#include "sawe/project.h"

#include <QWidget>

class QAbstractTableModel;

namespace Ui {
    class TransformInfoForm;
}

namespace Tools {

class TransformInfoForm : public QWidget
{
    Q_OBJECT

public:
    TransformInfoForm(Sawe::Project* project, RenderController* rendercontroller);
    ~TransformInfoForm();

public slots:
    void transformChanged();

private:
    Ui::TransformInfoForm *ui;
    QAbstractTableModel *model;
    Sawe::Project* project;
    RenderController* rendercontroller;

    QDockWidget* dock;

    void addRow(QString name, QString value);

};

} // namespace Tools

#endif // TRANSFORMINFOFORM_H
