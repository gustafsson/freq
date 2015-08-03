#ifndef FLACFILE_H
#define FLACFILE_H

#include <TargetConditionals.h>
#ifndef TARGET_IPHONE_SIMULATOR
#include "signal/operation.h"
#include "signal/cache.h"
#include <QUrl>

struct FlacFormat;
class FlacFile: public Signal::OperationDesc
{
public:
    FlacFile(QUrl url);
    FlacFile(const FlacFile&) = delete;
    FlacFile& operator=(const FlacFile&) = delete;
    ~FlacFile();

    Signal::Interval requiredInterval( const Signal::Interval& I, Signal::Interval* expectedOutput ) const override;
    Signal::Interval affectedInterval( const Signal::Interval& I ) const override;
    Signal::OperationDesc::ptr copy() const override;
    Signal::Operation::ptr createOperation(Signal::ComputingEngine* engine=0) const override;
    Extent extent() const override;
    QString toString() const override;
    bool operator==(const OperationDesc& d) const override;

private:
    QUrl url;
    void *decoderp;

    FlacFormat* fmt;
};

#endif
#endif // FLACFILE_H
