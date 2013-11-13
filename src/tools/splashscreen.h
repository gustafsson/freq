#ifndef SPLASHSCREEN_H
#define SPLASHSCREEN_H

#include <QDialog>
#include <QMap>

namespace Tools {

namespace Ui {
    class SplashScreen;
}

class SplashScreen : public QDialog
{
    Q_OBJECT

public:
    explicit SplashScreen(QWidget *parent = 0);
    ~SplashScreen();

private slots:
    void tickLoader();

private:
    Ui::SplashScreen *ui;
    QMap<float,QString> labels;
};


} // namespace Tools
#endif // SPLASHSCREEN_H
