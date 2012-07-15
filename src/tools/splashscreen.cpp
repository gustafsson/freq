#include "splashscreen.h"
#include "ui_splashscreen.h"

#include <math.h>

#include <QTimer>
#include <QResource>
#include <QSettings>

namespace Tools {

SplashScreen
        ::SplashScreen(QWidget *parent)
            :
    QDialog(parent),
    ui(new Ui::SplashScreen)
{
    ui->setupUi(this);

    setWindowFlags(Qt::FramelessWindowHint);

    QString splashImage;
    if (QSettings().value("target","") == "brickwall")
    {
        splashImage = ":/splash/brickwall.jpg";
        labels[0.f] = "Please wait while Brickwall Audio is taking over the world...";
        labels[.6f] = "Establishing power through brick building";
        labels[1.f] = "World domination complete";
    }
    else
    {
        splashImage = ":/splash/muchdifferent.jpg";
        labels[0.f] = "Please wait while Sonic AWE is loading..";
        labels[1.f] = "Launching Sonic AWE";
    }

    ui->progressBar->setValue(0);
    ui->progressBar->setMaximum(100);
    ui->labelSplash->setPixmap(QPixmap(splashImage));

    tickLoader();

    show();
}


SplashScreen::
        ~SplashScreen()
{
    delete ui;
}


void SplashScreen::
        tickLoader()
{
    if (ui->progressBar->value() >= ui->progressBar->maximum())
    {
        close();
        return;
    }

    ui->progressBar->setValue( ui->progressBar->value() + 1 );

    float v = (ui->progressBar->value() - ui->progressBar->minimum())
            / (float)(ui->progressBar->maximum()  - ui->progressBar->minimum());

    QMapIterator<float,QString> i(labels);
    QString last;
    while (i.hasNext()) {
        i.next();
        if (i.key() <= v)
            last = i.value();
    }
    if (ui->labelText->text() != last)
        ui->labelText->setText(last);

    this->setWindowOpacity(0.9 + 0.1*fabs(fmodf(v*10, 2)-1));

    bool isLast = 1.f > v;
    QTimer::singleShot( isLast ? 10 : 400, this, SLOT(tickLoader()));
}

} // namespace Tools
