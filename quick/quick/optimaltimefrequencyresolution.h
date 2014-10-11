#ifndef OPTIMALTIMEFREQUENCYRESOLUTION_H
#define OPTIMALTIMEFREQUENCYRESOLUTION_H

#include <QQuickItem>
#include "squircle.h"

class OptimalTimeFrequencyResolution : public QQuickItem
{
    Q_OBJECT
    Q_PROPERTY(Squircle* squircle READ squircle WRITE setSquircle NOTIFY squircleChanged)
public:
    explicit OptimalTimeFrequencyResolution(QQuickItem *parent = 0);

    Squircle* squircle() const { return squircle_; }
    void setSquircle(Squircle*s);

signals:
    void squircleChanged();

public slots:
    void onCameraChanged();

private:
    Squircle* squircle_ = 0;
};

#endif // OPTIMALTIMEFREQUENCYRESOLUTION_H
