#ifndef FLACFILE_H
#define FLACFILE_H

#include "signal/operation.h"
#include "signal/cache.h"
#include <QUrl>

class FlacFile: public Signal::OperationDesc
{
public:
    FlacFile(QUrl url);

    Signal::Interval requiredInterval( const Signal::Interval& I, Signal::Interval* expectedOutput ) const override;
    Signal::Interval affectedInterval( const Signal::Interval& I ) const override;
    Signal::OperationDesc::ptr copy() const override;
    Signal::Operation::ptr createOperation(Signal::ComputingEngine* engine=0) const override;
    Extent extent() const override;
    QString toString() const override;
    bool operator==(const OperationDesc& d) const override;

private:
    QUrl url;
    Signal::Cache data;
};

#endif // FLACFILE_H
