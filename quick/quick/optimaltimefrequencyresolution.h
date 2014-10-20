#ifndef OPTIMALTIMEFREQUENCYRESOLUTION_H
#define OPTIMALTIMEFREQUENCYRESOLUTION_H

#include <QQuickItem>
#include "squircle.h"

class OptimalTimeFrequencyResolution : public QQuickItem
{
    Q_OBJECT
    Q_PROPERTY(Squircle* squircle READ squircle WRITE setSquircle NOTIFY squircleChanged)
    Q_PROPERTY(bool paused READ isPaused WRITE setPaused NOTIFY pausedChanged)
public:
    explicit OptimalTimeFrequencyResolution(QQuickItem *parent = 0);

    Squircle* squircle() const { return squircle_; }
    void setSquircle(Squircle*s);

    bool isPaused() const { return paused_; }
    void setPaused(bool v);

signals:
    void squircleChanged();
    void pausedChanged();

public slots:
    void onCameraChanged();

private:
    Squircle* squircle_ = 0;
    bool paused_ = false;
};

#endif // OPTIMALTIMEFREQUENCYRESOLUTION_H
