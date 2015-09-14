#ifndef HEIGHTMAP_UPDATE_UPDATECONSUMER_H
#define HEIGHTMAP_UPDATE_UPDATECONSUMER_H

#include "updatequeue.h"
#include <QThread>

class QGLWidget;
class QOpenGLContext;
class QOffscreenSurface;

namespace Heightmap {
namespace Update {

class UpdateConsumerPrivate;
/**
 * @brief The UpdateConsumer class should update textures.
 *
 * UpdateConsumer is used by UpdateConsumerThread to do the actual work.
 *
 * UpdateConsumer can also be used from the render thread.
 */
class UpdateConsumer final
{
public:
    UpdateConsumer(UpdateQueue::ptr update_queue);
    ~UpdateConsumer();

    bool workIfAny();
    void blockUntilWork();

private:
    std::unique_ptr<UpdateConsumerPrivate> p;
};


/**
 * @brief The UpdateConsumerThread class should update textures in a thread
 * separate from both workers and rendering.
 */
class UpdateConsumerThread: public QThread
{
    Q_OBJECT
public:
    UpdateConsumerThread(QGLWidget* parent_and_shared_gl_context, UpdateQueue::ptr update_queue);
    UpdateConsumerThread(QOpenGLContext* shared_opengl_context, UpdateQueue::ptr update_queue, QObject* parent);
    ~UpdateConsumerThread();

signals:
    void didUpdate();

private slots:
    void threadFinished();

private:
    QOffscreenSurface* surface = 0;
    QOpenGLContext* shared_opengl_context = 0;
    UpdateQueue::ptr update_queue;

    void        run();

public:
    static void test();
};

} // namespace Update
} // namespace Heightmap

#endif // HEIGHTMAP_UPDATE_UPDATECONSUMER_H
