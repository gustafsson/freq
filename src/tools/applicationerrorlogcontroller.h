#ifndef TOOLS_APPLICATIONERRORLOGCONTROLLER_H
#define TOOLS_APPLICATIONERRORLOGCONTROLLER_H

#include <QToolBar>
#include <QPointer>
#include <QNetworkReply>
#include <QThread>

#include <boost/exception/all.hpp>

namespace Tools {

class SendFeedbackDialog;

namespace Support { class SendFeedback; }

/**
 * @brief The ApplicationErrorLogController class should collect information
 * about crashes to send anonymous feedback.
 */
class ApplicationErrorLogController : public QObject
{
    Q_OBJECT
public:
    static void registerException(boost::exception_ptr x);

    static void registerMainWindow(QMainWindow* window);

signals:
    void got_exception(boost::exception_ptr x);

private slots:
    // 'this' is owned by a separate thread, so logging takes place separately.
    void log(boost::exception_ptr x);
    void finishedSending(QNetworkReply*);

private:
    friend class OpenFeedbackDialog;

    ApplicationErrorLogController();
    static ApplicationErrorLogController* instance();

    std::list<QPointer<QToolBar> >      toolbars_;
    QThread                             thread_;
    Support::SendFeedback*              send_feedback_;
    class OpenFeedbackDialog*           open_feedback_dialog_;

public:
    static void test();
};

class OpenFeedbackDialog : public QObject
{
    Q_OBJECT
public:
    OpenFeedbackDialog();

public slots:
    void open();

private:
    SendFeedbackDialog*                 send_feedback_dialog_;
};


} // namespace Tools

#endif // TOOLS_APPLICATIONERRORLOGCONTROLLER_H
