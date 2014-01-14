#ifndef TFR_TRANSFORMOPERATION_H
#define TFR_TRANSFORMOPERATION_H

#include "signal/operation.h"

namespace Tfr {

class ChunkFilterDesc;
class TransformDesc;

/**
 * @brief The TransformOperationDesc class should wrap all generic functionality
 * in Signal::Operation and Tfr::Transform so that ChunkFilters can explicilty do
 * only the filtering.
 */
class TransformOperationDesc final: public Signal::OperationDesc
{
public:
    TransformOperationDesc(boost::shared_ptr<TransformDesc>, boost::shared_ptr<volatile ChunkFilterDesc>);
    ~TransformOperationDesc() {}

    // OperationDesc
    OperationDesc::Ptr copy() const;
    Signal::Operation::Ptr createOperation(Signal::ComputingEngine* engine=0) const;
    Signal::Interval requiredInterval(const Signal::Interval&, Signal::Interval*) const;
    Signal::Interval affectedInterval(const Signal::Interval&) const;
    Extent extent() const;
    QString toString() const;
    bool operator==(const Signal::OperationDesc&d) const;

    boost::shared_ptr<TransformDesc> transformDesc() const;
    virtual void transformDesc(boost::shared_ptr<TransformDesc> d);

    boost::shared_ptr<volatile ChunkFilterDesc> chunk_filter() const;

protected:
    boost::shared_ptr<volatile ChunkFilterDesc> chunk_filter_;

public:
    static void test();
};

} // namespace Tfr

#endif // TFR_TRANSFORMOPERATIONDESC_H
