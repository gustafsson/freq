#ifndef SELECTIONVIEWINFO_H
#define SELECTIONVIEWINFO_H

#include <QDockWidget>
#include <QPointer>

#include "signal/target.h"

namespace Sawe { class Project; }
namespace Tools {

namespace Ui {
    class SelectionViewInfo;
}

class SelectionModel;

class SelectionViewInfo : public QDockWidget
{
    Q_OBJECT

public:
    explicit SelectionViewInfo(Sawe::Project* project, SelectionModel* model);
    ~SelectionViewInfo();


public slots:
    void setText(QString text);
    void selectionChanged();

private:
    Sawe::Project* project_;
    QPointer<SelectionModel> model_;
    Ui::SelectionViewInfo *ui_;
    Signal::pTarget target_;
};


class SelectionViewInfoOperation: public Signal::Operation
{
public:
    SelectionViewInfoOperation( Signal::pOperation, SelectionViewInfo* info );

    virtual Signal::pBuffer read( const Signal::Interval& I );

private:
    SelectionViewInfo* info_;
    Signal::pOperation rms_;
};

} // namespace Tools
#endif // SELECTIONVIEWINFO_H
