#ifndef CHAIN_H
#define CHAIN_H

#include <QQuickItem>
#include <QOpenGLContext>
#include "signal/processing/chain.h"
#include "signal/recorder.h"
#include "heightmap/update/updatequeue.h"

class Chain : public QQuickItem
{
    Q_OBJECT
public:
    explicit Chain(QQuickItem *parent = 0);
    ~Chain();

    Signal::Processing::Chain::ptr chain() const { return chain_; }
    Signal::Processing::TargetMarker::ptr target_marker() const { return target_marker_; }
    Heightmap::Update::UpdateQueue::ptr update_queue() const { return update_queue_; }

signals:

private slots:
    void handleWindowChanged(QQuickWindow*);
    void clearOpenGlBackground();
    void setupUpdateConsumer(QOpenGLContext* context);

private:
    void openRecording();

    Signal::Recorder::ptr rec;
    Signal::Processing::Chain::ptr chain_;
    Signal::Processing::TargetMarker::ptr target_marker_;
    Heightmap::Update::UpdateQueue::ptr update_queue_;
    QPointer<QObject> update_consumer_=0;

};

#endif // CHAIN_H
