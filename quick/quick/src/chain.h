#ifndef CHAIN_H
#define CHAIN_H

#include "gl.h"
#include "signal/processing/chain.h"
#include "heightmap/update/updatequeue.h"
#include "heightmap/update/updateconsumer.h"
#include "timer.h"
#include "logtickfrequency.h"
#include "signal/recorder.h"

#include <QQuickItem>
#include <QOpenGLContext>

class Chain : public QQuickItem
{
    Q_OBJECT
    Q_PROPERTY(QString title READ title WRITE setTitle NOTIFY titleChanged)
public:
    explicit Chain(QQuickItem *parent = 0);
    ~Chain();

    Signal::Processing::Chain::ptr chain() const { return chain_; }
    Signal::Processing::TargetMarker::ptr target_marker() const { return target_marker_; }
    Heightmap::Update::UpdateQueue::ptr update_queue() const { return update_queue_; }

    QString title() { return title_; }
    void setTitle(QString v);

signals:
    void titleChanged();

private slots:
    void handleWindowChanged(QQuickWindow*);
    void clearOpenGlBackground();
    void setupBedroomUpdateThread();
    void setupUpdateConsumerThread(QOpenGLContext* context);
    void afterRendering();
    void sceneGraphInitialized();
    void sceneGraphInvalidated();

private:
    void openRecording();

    QString title_;
    Signal::Processing::Chain::ptr chain_;
    Signal::Recorder::ptr::weak_ptr rec_;
    Signal::Processing::TargetMarker::ptr target_marker_;
    Heightmap::Update::UpdateQueue::ptr update_queue_;
    QPointer<QObject> update_consumer_thread_=0;
    std::unique_ptr<Heightmap::Update::UpdateConsumer> update_consumer_p;
#ifndef LEGACY_OPENGL
    unsigned vertexArray_ = 0;
#endif

    LogTickFrequency ltf;
    Timer start_timer;
    Timer render_timer;
    double render_time = 0.;
};

#endif // CHAIN_H
