#ifndef SIGNAL_PROCESSING_PURGE_H
#define SIGNAL_PROCESSING_PURGE_H

#include "dag.h"
#include "targetneeds.h"

namespace Signal {
namespace Processing {

/**
 * @brief The Purge class should release allocated cache blocks that aren't
 * needed.
 */
class Purge
{
public:
    Purge(Dag::ptr::weak_ptr dag);
    
    void purge(TargetNeeds::ptr needs);
    
private:
    Dag::ptr::weak_ptr dag;
};

} // namespace Processing
} // namespace Signal

#endif // SIGNAL_PROCESSING_PURGE_H
