#ifndef RECORDVIEW_H
#define RECORDVIEW_H

#include <QObject>
#include "signal/worker.h"

namespace Tools
{

class RecordModel;

class RecordView: public QObject
{
    Q_OBJECT
public:
    RecordView(RecordModel* model);
    ~RecordView();

    bool enabled;

public slots:
    /// Connected in RecordController
    virtual void draw();

private:
    friend class RecordController;
    RecordModel* model_;
};

} // namespace Tools

#endif // RECORDVIEW_H
