#ifndef BLOCKUPDATER_H
#define BLOCKUPDATER_H

#include "glprojection.h"
#include "shared_state.h"

#include <functional>
#include <list>
#include <boost/shared_ptr.hpp>

namespace Heightmap {
class Block;

namespace BlockManagement {
class Fbo2Block;


/**
 * @brief The BlockUpdater class is safe to use from multiple threads.
 */
class BlockUpdater
{
public:
    typedef std::shared_ptr<BlockUpdater> Ptr;
    typedef std::function<bool(const glProjection& M)> DrawFunc;
    typedef boost::shared_ptr<Heightmap::Block> pBlock;

    BlockUpdater();
    BlockUpdater(const BlockUpdater&)=delete;
    BlockUpdater& operator=(const BlockUpdater&)=delete;
    ~BlockUpdater();

    void processUpdates(bool isMainThread);

    /**
     * @brief queueUpdate puts f in a queue. 'f(M)' will be called when 'b' is
     * mapped to a framebuffer object and M describes a orgthographic
     * projection onto the block (with the first axis time, the second
     * normalized scale).
     * @param f
     */
    void queueUpdate(pBlock b, DrawFunc && f);

private:
    shared_state<std::list<std::pair<pBlock, DrawFunc>>> queue_;
    Heightmap::BlockManagement::Fbo2Block* fbo2block;
};


} // namespace BlockManagement
} // namespace Heightmap

#endif // BLOCKUPDATER_H
