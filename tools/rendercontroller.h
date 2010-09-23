#ifndef RENDERCONTROLLER_H
#define RENDERCONTROLLER_H

#include "renderview.h"
#include <QWidget>

class QToolBar;
class QSlider;
namespace Ui { class ComboBoxAction; }

namespace Tools
{
    class RenderController: public QObject
    {
        Q_OBJECT
    public:
        RenderController( RenderView *view );
        ~RenderController();

    public slots:
        // GUI bindings are set up in RenderController constructor

        // ComboBoxAction color
        void receiveSetRainbowColors();
        void receiveSetGrayscaleColors();

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

    private:
        RenderModel *model;
        RenderView *view;

        // These are never used outside setupGui, but they are named here
        // to make it clear what class that is responsible for them.
        QToolBar* toolbar_render;
        Ui::ComboBoxAction* hzmarker;
        Ui::ComboBoxAction* color;
        Ui::ComboBoxAction* transform;
        QSlider * yscale;
        QSlider * tf_resolution;

        void setupGui();

    private slots:
        void clearCachedHeightmap();

    };
} // namespace Tools

#endif // RENDERCONTROLLER_H
