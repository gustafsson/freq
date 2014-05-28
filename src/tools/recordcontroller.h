#ifndef RECORDCONTROLLER_H
#define RECORDCONTROLLER_H

#include <QObject>

#include "recordview.h"

class QAction;

namespace Tools
{
    class RecordModel;

    /**
     * @brief The RecordController class should map UI actions to manage a recorder operation (RecordModel).
     */
    class RecordController: public QObject
    {
        Q_OBJECT
    public:
        RecordController( RecordView* view, QAction* actionRecord );
        ~RecordController();

    protected slots:
        void destroying();
        void receiveRecord(bool);
        void receiveStop();
        void redraw(Signal::Interval);

    private:
        // Model
        RecordView* view_;
        RecordModel* model() { return view_->model_; }

        struct Actions {
            QAction* actionRecord;
        };
        boost::shared_ptr<Actions> ui;

        bool destroyed_;
        Signal::IntervalType prev_num_samples_;

        void setupGui();
    };
} // namespace Tools
#endif // RECORDCONTROLLER_H
