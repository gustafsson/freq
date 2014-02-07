#ifndef RECORDVIEW_H
#define RECORDVIEW_H

#include "recordmodel.h"

#include <QObject>

namespace Tools
{

class RecordView: public QObject
{
    Q_OBJECT
public:
    RecordView(RecordModel* model);
    ~RecordView();

    void setEnabled(bool v);

signals:
    void gotNoData();

public slots:
    /// Connected in RecordController
    virtual void prePaint();

private:
    friend class RecordController;
    bool enabled_;
    RecordModel* model_;
    double prev_limit_;
};

} // namespace Tools

#endif // RECORDVIEW_H
