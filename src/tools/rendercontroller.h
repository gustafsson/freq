#ifndef RENDERCONTROLLER_H
#define RENDERCONTROLLER_H

#include "renderview.h"

#include <QWidget>
#include <QPointer>

class QToolBar;
class QSlider;
class QToolButton;

namespace Tfr { class Transform; }

namespace Ui { class ComboBoxAction; class MainWindow; }

namespace Tools
{
    namespace Widgets { class ValueSlider; }

    class SaweDll RenderController: public QObject // This should not be a QWidget. User input is handled by tools.
    {
        Q_OBJECT
    public:
        RenderController( QPointer<RenderView> view );
        ~RenderController();

        RenderModel*        model();

        ::Ui::MainWindow*   getItems();
        void                setBlockFilter(Heightmap::MergeChunkDesc::ptr mcdp, Tfr::TransformDesc::ptr transform_desc);
        void                stateChanged();
        void                emitAxisChanged();

        Ui::ComboBoxAction* transform;
        Ui::ComboBoxAction* hz_scale;
        Ui::ComboBoxAction* amplitude_scale;
        Ui::ComboBoxAction* hzmarker;
        QAction*            hz_scale_action;
        QAction*            amplitude_scale_action;
        QAction*            tf_resolution_action;
        QAction*            waveformScale;
        QAction*            linearScale;

    public slots:
        // GUI bindings are set up in RenderController constructor

        // ComboBoxAction color
        void receiveSetRainbowColors();
        void receiveSetGrayscaleColors();
        void receiveSetBlackGrayscaleColors();
        void receiveSetColorscaleColors();
        void receiveSetGreenRedColors();
        void receiveSetGreenWhiteColors();
        void receiveSetGreenColors();

        // Toggle Buttons
        void receiveToogleHeightlines(bool);
        void receiveToggleOrientation(bool);

        // ComboBoxAction hzmarker
        void receiveTogglePiano(bool);
        void receiveToggleHz(bool);
        void receiveToggleTAxis(bool);
        void receiveToggleCursorMarker(bool);

        // Sliders
        void receiveSetYScale(qreal);
        void receiveSetYBottom(qreal);
        void receiveSetTimeFrequencyResolution(qreal);
        void yscaleIncrease();
        void yscaleDecrease();
        void tfresolutionIncrease();
        void tfresolutionDecrease();

        // ComboBoxAction transform
        void receiveSetTransform_Cwt();
        void receiveSetTransform_Stft();
        void receiveSetTransform_Cwt_phase();
        void receiveSetTransform_Cwt_weight();
        void receiveSetTransform_Cepstrum();

        // ComboBoxAction hz_scale
        void receiveWaveformScale();
        void receiveLinearScale();
        void receiveLogScale();
        void receiveCepstraScale();

        // ComboBoxAction amplitude_scale
        void receiveLinearAmplitude();
        void receiveLogAmplitude();
        void receiveFifthAmplitude();

        void transformChanged();
        void updateTransformDesc();

    private slots:
        void deleteTarget();
        void updateFreqAxis();
        void updateAmplitudeAxis();
        void updateChannels();
        void reroute();
        void clearCaches();

    private:
        void setCurrentFilterTransform(Tfr::TransformDesc::ptr);
        void setBlockFilter(Tfr::ChunkFilterDesc::ptr kernel);
        //Tfr::Filter* currentFilter();
        Tfr::TransformDesc::ptr currentTransform();
        float headSampleRate();
        float currentTransformMinHz();

        QPointer<RenderView> view;

        // GUI stuff
        // These are never used outside setupGui, but they are named here
        // to make it clear what class that is "responsible" for them.
        // By responsible I mean to create them, insert them to their proper
        // place in the GUI and take care of events. The objects lifetime
        // depends on the parent QObject which they are inserted into.
        QToolBar* toolbar_render;
        QAction* logScale;
        QAction* cepstraScale;
        Ui::ComboBoxAction* color;
        QToolButton* channelselector;

        Widgets::ValueSlider* yscale;
        Widgets::ValueSlider* ybottom;
        Widgets::ValueSlider* tf_resolution;

        void setupGui();
        void windowLostFocus();
        void windowGotFocus();
        virtual bool eventFilter(QObject*, QEvent*);

        void toolbarWidgetVisible(QWidget*, bool);
        static void toolbarWidgetVisible(QToolBar*, QWidget*, bool);
    };
} // namespace Tools

#endif // RENDERCONTROLLER_H
