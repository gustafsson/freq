#ifndef RECORDCONTROLLER_H
#define RECORDCONTROLLER_H

#include <QObject>

#include "recordview.h"

namespace Ui { class MainWindow; }
namespace Adapters { class MicrophoneRecorder; }

namespace Tools
{
    class RecordModel;

    class RecordController: public QObject
    {
        Q_OBJECT
    public:
        RecordController( RecordView* view );
        ~RecordController();

    protected slots:
        void destroying();
        void receiveRecord(bool);
        void receiveStop();
        void recievedInvalidSamples( Signal::Intervals I );

    private:
        // Model
        RecordView* view_;
        RecordModel* model() { return view_->model_; }
        bool destroyed_;
        Signal::IntervalType prev_length_;

        void setupGui();
    };
} // namespace Tools
#endif // RECORDCONTROLLER_H
