#ifndef SENDFEEDBACK_H
#define SENDFEEDBACK_H

#include <QDialog>
#include <QScopedPointer>

class QNetworkReply;
class QNetworkAccessManager;

namespace Ui {
    class SaweMainWindow;
}

namespace Tools {

namespace Ui {
    class SendFeedback;
}

class SendFeedback : public QDialog
{
    Q_OBJECT

public:
    explicit SendFeedback(::Ui::SaweMainWindow *parent);
    ~SendFeedback();

    virtual void accept();

private slots:
    void browse();
    void replyFinished(QNetworkReply*);

private:
    void sendLogFiles(QString email, QString message, QString extraFile);

    Ui::SendFeedback *ui;
    QString targetUrl;
    QScopedPointer<QNetworkAccessManager> manager;
};


} // namespace Tools
#endif // SENDFEEDBACK_H
