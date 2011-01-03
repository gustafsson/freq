#ifndef RECORDCONTROLLER_H
#define RECORDCONTROLLER_H

#include <QObject>
#include <signal/operation.h>

namespace Ui { class MainWindow; }
namespace Signal { class Worker; }
namespace Adapters { class MicrophoneRecorder; }

namespace Tools
{
    class RenderView;

    class RecordController: QObject
    {
        Q_OBJECT
    public:
        RecordController( Signal::Worker* worker, Ui::MainWindow* actions );
        ~RecordController();

    protected slots:
        void receiveRecord(bool);
        void recievedBuffer(Signal::pBuffer);

    private:
        // Model
        Signal::pOperation _record_model;
        Signal::Worker* _worker;

        void setupGui( Ui::MainWindow* actions );
    };
} // namespace Tools
#endif // RECORDCONTROLLER_H
