#ifndef TOOLS_SUGGESTPURCHASE_H
#define TOOLS_SUGGESTPURCHASE_H

#include <QWidget>

namespace Tools {

class SuggestPurchase : public QObject
{
    Q_OBJECT
public:
    explicit SuggestPurchase(QWidget *mainWindow);

signals:

private slots:
    void suggest();

private:
    void dropnotify();
};

} // namespace Tools

#endif // TOOLS_SUGGESTPURCHASE_H
