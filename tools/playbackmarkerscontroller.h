#ifndef PLAYBACKMARKERSCONTROLLER_H
#define PLAYBACKMARKERSCONTROLLER_H

#include "playbackmarkersview.h"

#include <QWidget>

namespace Tools {

class PlaybackMarkersController: public QWidget
{
    Q_OBJECT
public:
    PlaybackMarkersController(PlaybackMarkersView* view);

public slots:
    void addMarker();

private:
    PlaybackMarkersView* view_;
};

} // namespace Tools

#endif // PLAYBACKMARKERSCONTROLLER_H
