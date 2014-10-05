#ifndef CHAIN_H
#define CHAIN_H

#include <QQuickItem>
#include "signal/processing/chain.h"
#include "signal/recorder.h"

class Chain : public QQuickItem
{
    Q_OBJECT
public:
    explicit Chain(QQuickItem *parent = 0);

    Signal::Processing::Chain::ptr chain() const { return chain_; }
    Signal::Processing::TargetMarker::ptr target_marker() const { return target_marker_; }

signals:

private slots:
    void handleWindowChanged(QQuickWindow*);
    void clearOpenGlBackground();

private:
    void openRecording();

    Signal::Recorder::ptr rec;
    Signal::Processing::Chain::ptr chain_;
    Signal::Processing::TargetMarker::ptr target_marker_;
    QObject* background_clearer_=0;
};

#endif // CHAIN_H
