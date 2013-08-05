#ifndef SELECTIONVIEWINFO_H
#define SELECTIONVIEWINFO_H

#include <QDockWidget>
#include <QPointer>

#include "signal/target.h"
#include "signal/processing/targetmarker.h"

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
    void append(QString text);
    void selectionChanged();
    virtual void setVisible(bool visible);

private:
    Sawe::Project* project_;
    QPointer<SelectionModel> model_;
    Ui::SelectionViewInfo *ui_;
    Signal::pTarget target_;
//Use Signal::Processing namespace
    Signal::Processing::TargetMarker::Ptr target_marker_;
};


class SelectionViewInfoSink: public Signal::Sink
{
public:
    SelectionViewInfoSink( SelectionViewInfo* info );

    virtual Signal::pBuffer read( const Signal::Interval& I );

    virtual void source(Signal::pOperation v);

    virtual void invalidate_samples(const Signal::Intervals& I);
    virtual Signal::Intervals invalid_samples();

private:
    SelectionViewInfo* info_;
    Signal::pOperation rms_;
    Signal::pOperation mixchannels_;
    Signal::Intervals missing_;
    bool searchingformaximas_;
};

} // namespace Tools
#endif // SELECTIONVIEWINFO_H
