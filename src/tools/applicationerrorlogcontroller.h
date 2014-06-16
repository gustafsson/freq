#ifndef TOOLS_APPLICATIONERRORLOGCONTROLLER_H
#define TOOLS_APPLICATIONERRORLOGCONTROLLER_H

#include <QToolBar>
#include <QPointer>
#include <QNetworkReply>
#include <QThread>

#include <boost/exception_ptr.hpp>

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
    ~ApplicationErrorLogController();

    static void registerException(boost::exception_ptr x) noexcept;

    static void registerMainWindow(QMainWindow* window);

signals:
    void got_exception(boost::exception_ptr x);
    void showToolbar(bool v=false);

private slots:
    // 'this' is owned by a separate thread, so logging takes place separately.
    void log(boost::exception_ptr x);
    void finishedSending(QNetworkReply*);
    void finishedOk();

private:
    ApplicationErrorLogController();
    static ApplicationErrorLogController* instance();

    int                                 feedback_count_ = 0;
    int                                 feedback_limit_ = 1;
    QThread                             thread_;
    Support::SendFeedback*              send_feedback_ = 0;
    bool                                finished_ok_ = false;

public:
    static void test();
};

class OpenFeedbackDialog : public QObject
{
    Q_OBJECT
public:
    OpenFeedbackDialog(QWidget* parent, QToolBar* bar);

signals:
    void dialogOpened();

public slots:
    void showToolbar(bool v);
    void open();

private:
    QPointer<QToolBar>  bar_;
    SendFeedbackDialog* send_feedback_dialog_;
};


} // namespace Tools

#endif // TOOLS_APPLICATIONERRORLOGCONTROLLER_H
