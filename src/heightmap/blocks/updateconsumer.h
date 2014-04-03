#ifndef HEIGHTMAP_BLOCKS_UPDATECONSUMER_H
#define HEIGHTMAP_BLOCKS_UPDATECONSUMER_H

#include "blockupdater.h"

#include <QThread>

class QGLWidget;

namespace Heightmap {
namespace Blocks {

class UpdateQueue;

/**
 * @brief The UpdateConsumer class should update textures in a separate thread
 * from the worker thread.
 */
class UpdateConsumer: public QThread
{
    Q_OBJECT
public:
    UpdateConsumer(QGLWidget* parent_and_shared_gl_context, UpdateQueue::ptr update_queue);
    ~UpdateConsumer();

private slots:
    void threadFinished();

private:
    QGLWidget*       shared_gl_context;
    std::shared_ptr<UpdateQueue> update_queue;
    BlockUpdater     block_updater;

    void        run();

public:
    static void test();
};

} // namespace Blocks
} // namespace Heightmap

#endif // HEIGHTMAP_BLOCKS_UPDATECONSUMER_H
