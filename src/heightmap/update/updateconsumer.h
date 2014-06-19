#ifndef HEIGHTMAP_UPDATE_UPDATECONSUMER_H
#define HEIGHTMAP_UPDATE_UPDATECONSUMER_H

#include "updatequeue.h"
#include <QThread>

class QGLWidget;

namespace Heightmap {
namespace Update {

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

signals:
    void didUpdate();

private slots:
    void threadFinished();

private:
    QGLWidget*   shared_gl_context;
    UpdateQueue::ptr update_queue;

    void        run();

public:
    static void test();
};

} // namespace Update
} // namespace Heightmap

#endif // HEIGHTMAP_UPDATE_UPDATECONSUMER_H
