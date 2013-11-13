#ifndef CHECKUPDATES_H
#define CHECKUPDATES_H

#include <QObject>
#include <QScopedPointer>


class QNetworkReply;
class QNetworkAccessManager;

namespace Ui
{
    class SaweMainWindow;
}

namespace Tools {

class CheckUpdates : public QObject
{
    Q_OBJECT

public:
    explicit CheckUpdates(::Ui::SaweMainWindow *parent);
    virtual ~CheckUpdates();

signals:

private slots:
    void checkForUpdates();
    void autoCheckForUpdatesSoon();
    void autoCheckForUpdates();
    void replyFinished(QNetworkReply*);

private:
    bool manualUpdate;
    QString targetUrl, checkUpdatesTag;
    QScopedPointer<QNetworkAccessManager> manager;
};

} // namespace Tools

#endif // CHECKUPDATES_H
