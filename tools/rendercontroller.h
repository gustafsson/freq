#ifndef RENDERCONTROLLER_H
#define RENDERCONTROLLER_H

#include "renderview.h"

#include "signal/worker.h"

#include <QWidget>
#include <QPointer>

class QToolBar;
class QSlider;

namespace Ui { class ComboBoxAction; }

namespace Tools
{
    class RenderController: public QObject // This should not be a QWidget. User input is handled by tools.
    {
        Q_OBJECT
    public:
        RenderController( QPointer<RenderView> view );
        ~RenderController();

    public slots:
        // GUI bindings are set up in RenderController constructor

        // ComboBoxAction color
        void receiveSetRainbowColors();
        void receiveSetGrayscaleColors();
        void receiveSetColorscaleColors();

        // Toggle Buttons
        void receiveToogleHeightlines(bool);

        // ComboBoxAction hzmarker
        void receiveTogglePiano(bool);
        void receiveToggleHz(bool);

        // Sliders
        void receiveSetYScale(int);
        void receiveSetTimeFrequencyResolution(int);

        // ComboBoxAction transform
        void receiveSetTransform_Cwt();
        void receiveSetTransform_Stft();
        void receiveSetTransform_Cwt_phase();
        void receiveSetTransform_Cwt_reassign();
        void receiveSetTransform_Cwt_ridge();
        void receiveSetTransform_Cwt_weight();

    private slots:
        void clearCachedHeightmap();

    private:
        Signal::PostSink* setBlockFilter(Signal::Operation* blockfilter);

        RenderModel *model();
        QPointer<RenderView> view;

        // GUI stuff
        // These are never used outside setupGui, but they are named here
        // to make it clear what class that is "responsible" for them.
        // By responsible I mean to create them, insert them to their proper
        // place in the GUI and take care of events. The objects lifetime
        // depends on the parent QObject which they are inserted into.
        QToolBar* toolbar_render;
        Ui::ComboBoxAction* hzmarker;
        Ui::ComboBoxAction* color;
        Ui::ComboBoxAction* transform;
        QSlider * yscale;
        QSlider * tf_resolution;

        void setupGui();

        // Controlling
        Signal::pOperation _updateViewSink;
        Signal::pWorkerCallback _updateViewCallback;
    };
} // namespace Tools

#endif // RENDERCONTROLLER_H
