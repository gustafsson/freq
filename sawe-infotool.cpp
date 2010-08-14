#include "sawe-infotool.h"
#include <QToolTip>
#include <iostream>
#include <iomanip>

namespace Sawe{

void InfoTool::mousePressEvent(QMouseEvent * e)
{
    if( (e->button() & Qt::LeftButton) == Qt::LeftButton && !e->modifiers().testFlag(Qt::AltModifier))
    {
        usingInfo = true;
        mouseMoveEvent( e );
    }
    else
        e->ignore();
}

void InfoTool::mouseReleaseEvent(QMouseEvent * e)
{
    if( (e->button() & Qt::LeftButton) == Qt::LeftButton && usingInfo)
    {
        usingInfo = false;
    }
    else
        e->ignore();
}

void InfoTool::mouseMoveEvent(QMouseEvent * e)
{
    if(!usingInfo)
    {
        e->ignore();
        return;
    }
    
    printf("Infortool: \n");
    using namespace std;
    int x = e->x(), y = this->height() - e->y();
    GLvector current;
    if( displayWidget->worldPos(x, y, current[0], current[1], displayWidget->xscale) )
    {
        const Tfr::pCwt c = Tfr::CwtSingleton::instance();
        unsigned FS = displayWidget->worker()->source()->sample_rate();
        float t = ((unsigned)(current[0]*FS+.5f))/(float)FS;
        current[1] = ((unsigned)(current[1]*c->nScales(FS)+.5f))/(float)c->nScales(FS);
        float f = c->compute_frequency( current[1], FS );
        float std_t = c->morlet_std_t(current[1], FS);
        float std_f = c->morlet_std_f(current[1], FS);
    
        stringstream ss;
        ss << setiosflags(ios::fixed)
            << "Time: " << setprecision(3) << t << " s" << endl
            << "Frequency: " << setprecision(1) << f << " Hz" << endl
            << "Standard deviation: " << setprecision(3) << std_t << " s, " << setprecision(1) << std_f << " Hz";
        QToolTip::showText( e->globalPos(), QString::fromLocal8Bit("..."), this ); // Force tooltip to change position even if the text is the same as in previous tooltip
        QToolTip::showText( e->globalPos(), QString::fromLocal8Bit(ss.str().c_str()), this );
    }
}

};