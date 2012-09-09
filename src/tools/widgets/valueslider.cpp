#include "valueslider.h"

#include <QSlider>
#include <QDialog>
#include <QLineEdit>
#include <QEvent>

namespace Tools {
namespace Widgets {

ValueSlider::
        ValueSlider(QWidget *parent) :
    QComboBox(parent),
    popup_(0),
    slider_(0),
    resolution_(0)
{
    popup_ = new QDialog(this);
    slider_ = new QSlider(Qt::Horizontal, popup_);
    popup_->setWindowFlags(Qt::Popup);

    popup_->installEventFilter(this);

    // update layout
    popup_->setAttribute(Qt::WA_DontShowOnScreen,true);
    popup_->show();
    popup_->hide();
    popup_->setAttribute(Qt::WA_DontShowOnScreen,false);

    setEditable(true);

    connect(slider_, SIGNAL(valueChanged(int)), SLOT(valueChanged(int)));
    connect(slider_, SIGNAL(rangeChanged(int,int)), SLOT(rangeChanged(int,int)));
    connect(slider_, SIGNAL(sliderMoved(int)), SLOT(sliderMoved(int)));
    connect(lineEdit(), SIGNAL(editingFinished()), SLOT(editingFinished()));

    setDecimals(3);
    setMin(0);
    setMax(100);
    setValue(50);
}


ValueSlider::
        ~ValueSlider()
{
    delete popup_;
}


void ValueSlider::
        showPopup()
{
    popup_->move(mapToGlobal(QPoint(0,height())));
    popup_->show(); // hidden by WindowFlags Qt::Popup
}


void ValueSlider::
        hidePopup()
{
    QComboBox::hidePopup();
}


void ValueSlider::
        focusOutEvent ( QFocusEvent * e )
{
    QComboBox::focusOutEvent(e);
}


bool ValueSlider::
        eventFilter(QObject *o, QEvent *e)
{
    if (o == popup_ && e->type() == QEvent::Hide)
        hidePopup();

    return false;
}


qreal ValueSlider::
        min() const
{
    return toReal(slider_->minimum());
}


void ValueSlider::
        setMin(qreal m)
{
    slider_->setMinimum(toInt(m));
}


qreal ValueSlider::
        max() const
{
    return toReal(slider_->maximum());
}


void ValueSlider::
        setMax(qreal m)
{
    slider_->setMaximum(toInt(m));
}


qreal ValueSlider::
        value() const
{
    return toReal(slider_->value());
}


void ValueSlider::
        setValue(qreal v)
{
    slider_->setValue(toInt(v));
}



int ValueSlider::
        decimals() const
{
    int r = resolution_;
    int d = 0;
    while(r>0)
    {
        r/=10;
        d++;
    }
    return d;
}


void ValueSlider::
        setDecimals(int d)
{
    if (d > 10)
        d = 10;

    resolution_ = 1;
    while(d>0)
    {
        resolution_*=10;
        d--;
    }

    setValue(value());
    setMin(min());
    setMax(max());
}


void ValueSlider::
        valueChanged(int v)
{
    sliderMoved(v);
    emit valueChanged(toReal(v));
}


void ValueSlider::
        rangeChanged(int min, int max)
{
    emit rangeChanged(toReal(min), toReal(max));

    lineEdit()->setToolTip(QString("Enter a value between %1 and %2").arg(this->min()).arg(this->max()));
    setValidator(new QDoubleValidator(this->min(), this->max(), decimals()));
}


void ValueSlider::
        sliderMoved(int v)
{
    qreal r = toReal(v);
    lineEdit()->setText(QString("%1").arg(r));
    emit sliderMoved(r);
}


void ValueSlider::
        editingFinished()
{
    QString text = currentText();
    bool ok = false;
    qreal v = text.toDouble(&ok);

    if (!ok)
        v = value();

    setValue( v );
}


qreal ValueSlider::
        toReal(int v) const
{
    return qreal(v)/resolution_;
}


int ValueSlider::
        toInt(qreal v) const
{
    return v*resolution_ + 0.5;
}

} // namespace Support
} // namespace Tools
