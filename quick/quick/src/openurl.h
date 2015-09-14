#ifndef OPENURL_H
#define OPENURL_H

#include <QQuickItem>
#include "chain.h"

class OpenUrl : public QQuickItem
{
    Q_OBJECT
    Q_PROPERTY(Chain* chain READ chain WRITE setChain NOTIFY chainChanged)

public:
    explicit OpenUrl(QQuickItem *parent = 0);

    Chain* chain() const { return chain_; }
    void setChain(Chain* c);

signals:
    void chainChanged();
    void openFileInfo(QString infoText);

private slots:
    // bound to qml signal openUrl(QUrl)
    void onOpenUrl(QUrl url);

protected:
    void componentComplete() override;

    Chain* chain_;
};

#endif // OPENURL_H
