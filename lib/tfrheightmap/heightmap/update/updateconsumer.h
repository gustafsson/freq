#ifndef HEIGHTMAP_UPDATE_UPDATECONSUMER_H
#define HEIGHTMAP_UPDATE_UPDATECONSUMER_H

#include "updatequeue.h"
#include <QThread>

class QGLWidget;
class QOpenGLContext;
class QOffscreenSurface;

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
    UpdateConsumer(QOpenGLContext* shared_opengl_context, UpdateQueue::ptr update_queue, QObject* parent);
    ~UpdateConsumer();

signals:
    void didUpdate();

private slots:
    void threadFinished();

private:
    QOffscreenSurface* surface = 0;
    QOpenGLContext* shared_opengl_context = 0;
    UpdateQueue::ptr update_queue;

    void        run();
    void        work();

public:
    static void test();
};

} // namespace Update
} // namespace Heightmap

#endif // HEIGHTMAP_UPDATE_UPDATECONSUMER_H
