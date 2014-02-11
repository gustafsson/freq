#ifndef TOOLS_SUPPORT_SENDFEEDBACK_H
#define TOOLS_SUPPORT_SENDFEEDBACK_H

#include <QNetworkReply>
#include <QScopedPointer>

namespace Tools {
namespace Support {

class SendFeedback : public QObject
{
    Q_OBJECT
public:
    explicit SendFeedback(QObject *parent = 0);

    /**
     * @brief sendLogFiles
     * @param email
     * @param message
     * @param extraFile
     * @return A text string describing if any log files where omitted due to their size.
     */
    QString sendLogFiles(QString email, QString message, QString extraFile);

signals:
    void finished(QNetworkReply*);

private:
    QString targetUrl;
    QScopedPointer<QNetworkAccessManager> manager;
};

} // namespace Support
} // namespace Tools

#endif // TOOLS_SUPPORT_SENDFEEDBACK_H
