#ifndef TRANSFORMINFOFORM_H
#define TRANSFORMINFOFORM_H

#include "sawe/project.h"

#include <QWidget>
#include <QTimer>

class QAbstractTableModel;

namespace Ui {
    class TransformInfoForm;
}

namespace Tools {

class TransformInfoForm : public QWidget
{
    Q_OBJECT

public:
    TransformInfoForm(Sawe::Project* project, RenderView* renderview);
    virtual ~TransformInfoForm();

public slots:
    void transformChanged();
    void checkVisibility(bool);

    void minHzChanged();
    //void maxHzChanged();
    void binResolutionChanged();
    void windowSizeChanged();
    void sampleRateChanged();

private:
    Ui::TransformInfoForm *ui;
    QAbstractTableModel *model;
    Sawe::Project* project;
    RenderView* renderview;

    QDockWidget* dock;

    void addRow(QString name, QString value);

    QTimer timer;
};

} // namespace Tools

#endif // TRANSFORMINFOFORM_H
