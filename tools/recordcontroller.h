#ifndef RECORDCONTROLLER_H
#define RECORDCONTROLLER_H

#include <QObject>

#include "recordview.h"

namespace Ui { class MainWindow; }
namespace Adapters { class MicrophoneRecorder; }

namespace Tools
{
    class RecordModel;
    class RenderView;

    class RecordController: public QObject
    {
        Q_OBJECT
    public:
        RecordController( RecordView* view, RenderView* render_view );
        ~RecordController();

    protected slots:
        void stopRecording();
        void receiveRecord(bool);
        void recievedInvalidSamples( const Signal::Intervals& I );

    private:
        // Model
        RecordView* view_;
        RecordModel* model() { return view_->model_; }

        RenderView* render_view_;

        void setupGui();
    };
} // namespace Tools
#endif // RECORDCONTROLLER_H
