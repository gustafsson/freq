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

    bool enabled;

signals:
    void gotNoData();

public slots:
    /// Connected in RecordController
    virtual void prePaint();

private:
    friend class RecordController;
    RecordModel* model_;
    double prev_limit_;
};

} // namespace Tools

#endif // RECORDVIEW_H
