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
    TransformOperationDesc(shared_state<ChunkFilterDesc>);
    ~TransformOperationDesc() {}

    // OperationDesc
    OperationDesc::ptr copy() const;
    Signal::Operation::ptr createOperation(Signal::ComputingEngine* engine=0) const;
    Signal::Interval requiredInterval(const Signal::Interval&, Signal::Interval*) const;
    Signal::Interval affectedInterval(const Signal::Interval&) const;
    Extent extent() const;
    QString toString() const;
    bool operator==(const Signal::OperationDesc&d) const;

    boost::shared_ptr<TransformDesc>            transformDesc() const;
    void                                        transformDesc(boost::shared_ptr<TransformDesc>);
    shared_state<ChunkFilterDesc>::write_ptr    chunk_filter();
    shared_state<ChunkFilterDesc>::read_ptr     chunk_filter() const;

protected:
    shared_state<ChunkFilterDesc> chunk_filter_;
    boost::shared_ptr<TransformDesc> transformDesc_;

public:
    static void test();
};

} // namespace Tfr

#endif // TFR_TRANSFORMOPERATIONDESC_H
