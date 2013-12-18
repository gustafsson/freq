#ifndef TOOLS_WAVEFORMCONTROLLER_H
#define TOOLS_WAVEFORMCONTROLLER_H

#include <QObject>

#include "rendercontroller.h"

namespace Tools {

/**
 * @brief The WaveformController class should make the waveform accessible from the user interface.
 */
class WaveformController : public QObject
{
    Q_OBJECT
public:
    explicit WaveformController(Tools::RenderController* parent);

signals:

public slots:
    void                     receiveSetTransform_DrawnWaveform();

private:
    QAction*                 showWaveform;

    void                     setupGui();
    Tools::RenderController* render_controller();
};

} // namespace Tools

#endif // TOOLS_WAVEFORMCONTROLLER_H
