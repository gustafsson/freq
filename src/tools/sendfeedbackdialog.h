#ifndef TOOLS_SENDFEEDBACKDIALOG_H
#define TOOLS_SENDFEEDBACKDIALOG_H

#include <QDialog>

class QNetworkReply;
class QNetworkAccessManager;

namespace Ui {
    class SaweMainWindow;
}

namespace Tools {

namespace Ui {
    class SendFeedback;
}

namespace Support {
    class SendFeedback;
}

class SendFeedbackDialog : public QDialog
{
    Q_OBJECT
public:
    explicit SendFeedbackDialog(::Ui::SaweMainWindow *parent);
    ~SendFeedbackDialog();

    virtual void accept();

private slots:
    void browse();
    void replyFinished(QNetworkReply*);

private:
    Support::SendFeedback* sendfeedback;
    Ui::SendFeedback *ui;
};


} // namespace Tools
#endif // TOOLS_SENDFEEDBACKDIALOG_H
