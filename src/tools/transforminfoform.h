#ifndef TRANSFORMINFOFORM_H
#define TRANSFORMINFOFORM_H

#include "sawe/project.h"

#include <QWidget>
#include <QTimer>

class QLineEdit;

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
    //void sampleRateChanged();
    void windowTypeChanged();
    void overlapChanged();
    void averagingChanged();
    void timeNormalizationChanged(qreal);
    void freqNormalizationChanged(qreal);
    void freqNormalizationPercentChanged(qreal);

private:
    Ui::TransformInfoForm *ui;
    Sawe::Project* project;
    RenderView* renderview;

    QDockWidget* dock;

    void deprecateAll();
    void addRow(QString name, QString value);
    void setEditText(QLineEdit* edit, QString value);

    QTimer timer;
};

} // namespace Tools

#endif // TRANSFORMINFOFORM_H
