#include "valueslider.h"

#include <QSlider>
#include <QDialog>
#include <QLineEdit>
#include <QEvent>
#include <QApplication>
#include <QDesktopWidget>

#include <math.h>
#include <boost/math/special_functions/fpclassify.hpp>

namespace Tools {
namespace Widgets {

ValueSlider::
        ValueSlider(QWidget *parent) :
    QComboBox(parent),
    popup_(0),
    slider_(0),
    resolution_(1000000),
    decimals_(0),
    value_translation_(Linear),
    value_(0),
    slider_is_pressed_(false),
    logaritmic_zero_(false)
{
    setMinimumWidth (40);

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
    connect(slider_, SIGNAL(sliderPressed()), SLOT(sliderPressed()));
    connect(slider_, SIGNAL(sliderReleased()), SLOT(sliderReleased()));
    connect(lineEdit(), SIGNAL(editingFinished()), SLOT(editingFinished()));
    connect(lineEdit(), SIGNAL(returnPressed()), SLOT(returnPressed()));

    updateLineEditOnValueChanged (true);

    setDecimals(0);
    setMinimum(0);
    setMaximum(100);
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
    QDesktopWidget* desktop = QApplication::desktop();
    QRect currentScreen = desktop->screenGeometry (this);
    QPoint p = mapToGlobal(QPoint());

    QRect popup(p, popup_->size ());

    QRect belowLeftAlign = popup;
    belowLeftAlign.moveTop(p.y () + height());

    QRect belowRightAlign = belowLeftAlign;
    belowRightAlign.moveRight (p.x () + width());

    QRect aboveRightAlign = belowRightAlign;
    aboveRightAlign.moveBottom (p.y ());

    QRect aboveLeftAlign = belowLeftAlign;
    aboveLeftAlign.moveBottom (p.y ());

    QList<QRect> l;
    l.push_back (belowLeftAlign);
    l.push_back (belowRightAlign);
    l.push_back (aboveLeftAlign);
    l.push_back (aboveRightAlign);

    popup_->move (p);
    foreach(const QRect& t, l)
    {
        if (currentScreen.contains (t, true))
        {
            popup_->move (t.topLeft ());
            break;
        }
    }

    popup_->show(); // hidden afterwards by WindowFlags Qt::Popup
}


void ValueSlider::
        hidePopup()
{
    QComboBox::hidePopup();
}


void ValueSlider::
        focusInEvent ( QFocusEvent * e )
{
    QComboBox::focusInEvent(e);

    updateLineEdit();
}


void ValueSlider::
        focusOutEvent ( QFocusEvent * e )
{
    QComboBox::focusOutEvent(e);

    updateLineEdit();
}


bool ValueSlider::
        eventFilter(QObject *o, QEvent *e)
{
    if (o == popup_ && e->type() == QEvent::Hide)
        hidePopup();

    return false;
}


qreal ValueSlider::
        minimum() const
{
    return toReal(slider_->minimum());
}


void ValueSlider::
        setMinimum(qreal m)
{
    qreal max = maximum();
    if (max < m) max = m;
    setRange(m, max, value_translation_);
}


qreal ValueSlider::
        maximum() const
{
    return toReal(slider_->maximum());
}


void ValueSlider::
        setMaximum(qreal m)
{
    qreal min = minimum();
    if (min > m) min = m;
    setRange(min, m, value_translation_);
}


qreal ValueSlider::
        value() const
{
    return value_;
}


void ValueSlider::
        setValue(qreal v)
{
    qreal q = v;
    if (v < minimum())
        v = minimum();
    if (v > maximum())
        v = maximum();

    if (value_ == v && v == q)
        return;

    value_ = v;
    updateLineEdit ();

    if (!slider_is_pressed_)
        slider_->setValue(toInt(v));

    emit valueChanged(v);
}


void ValueSlider::
        setRange(qreal min, qreal max, ValueTranslation translation)
{
    qreal v = value_;

    qreal d;
    switch(translation)
    {
        case Quadratic:
            d = sqrt(max) - sqrt(min);
            break;
        case LogaritmicZeroMin:
            if (min <= 0) min = 1;
            // fall through
        case Logaritmic:
            if (max < min) max = min;

            d = log(max) - log(min);
            break;
        default:
            d = max - min;
            break;
    }
    if (boost::math::isnan(d) || boost::math::isinf(d) || d<100)
        d = 100;

    resolution_ = INT_MAX / d;
    value_translation_ = translation;
    slider_->setMinimum (toInt(min));
    slider_->setMaximum (toInt(max));
    slider_->setValue (toInt(v));
}


Qt::Orientation ValueSlider::
        orientation()
{
    return slider_->orientation ();
}


void ValueSlider::
        setOrientation( Qt::Orientation orientation )
{
    if (orientation != this->orientation())
    {
        QSize sz = slider_->size ();
        slider_->setOrientation ( orientation );
        sz.transpose ();
        slider_->resize (sz);
    }
}


QString ValueSlider::
        unit() const
{
    return unit_;
}


void ValueSlider::
        setUnit(QString value)
{
    unit_ = value;
    updateLineEdit();
}


QString ValueSlider::
        toolTip()
{
    return slider_->toolTip ();
}


void ValueSlider::
        setToolTip( QString str )
{
    slider_->setToolTip ( str );
    rangeChanged(slider_->minimum (), slider_->maximum ());
}


void ValueSlider::
        triggerAction ( QAbstractSlider::SliderAction action )
{
    qreal d = slider_->maximum () - slider_->minimum ();
    qreal f = slider_->value () - slider_->minimum ();

    f /= d;

    switch(action)
    {
    case QAbstractSlider::SliderNoAction:       break;
    case QAbstractSlider::SliderSingleStepAdd:  f+=0.01; break;
    case QAbstractSlider::SliderSingleStepSub:  f-=0.01; break;
    case QAbstractSlider::SliderPageStepAdd:    f+=0.1; break;
    case QAbstractSlider::SliderPageStepSub:    f-=0.1; break;
    case QAbstractSlider::SliderToMinimum:      f=0; break;
    case QAbstractSlider::SliderToMaximum:      f=1; break;
    case QAbstractSlider::SliderMove:           break;
    }

    if (f<0) f = 0;
    if (f>1) f = 1;

    f *= d;

    slider_->setValue (slider_->minimum () + f);

    sliderMoved(slider_->value ());
}


ValueSlider::ValueTranslation ValueSlider::
        valueTranslation()
{
    return value_translation_;
}


int ValueSlider::
        decimals() const
{
    return decimals_;
}


void ValueSlider::
        setDecimals(int d)
{
    decimals_ = d;

    lineEdit()->setText(QString("%1").arg(value_,0,'f',decimals(value_)));
    lineEdit()->setToolTip(QString("%3 [%1, %2]")
                           .arg(minimum(),0,'f',decimals(minimum()))
                           .arg(maximum(),0,'f',decimals(maximum()))
                           .arg(toolTip()));
}


void ValueSlider::
        setSliderSize(int s)
{
    QSize sz( s, 22 );
    if (orientation() == Qt::Vertical)
        sz.transpose ();
    slider_->resize ( sz );
}


void ValueSlider::
        updateLineEditOnValueChanged(bool v)
{
    if (v)
        connect(this, SIGNAL(valueChanged(qreal)), this, SLOT(updateLineEdit()));
    else
        disconnect(this, SIGNAL(valueChanged(qreal)), this, SLOT(updateLineEdit()));
}


void ValueSlider::
        valueChanged(int)
{
    // don't do anything if the value is programmatically set on the slider. SIGNAL(valueChanged(qreal)) is emitted in setValue and sliderMoved instead
}


void ValueSlider::
        rangeChanged(int min, int max)
{
    emit rangeChanged(toReal(min), toReal(max));

    lineEdit()->setToolTip(QString("%3 [%1, %2]")
                           .arg(minimum(),0,'f',decimals(minimum()))
                           .arg(maximum(),0,'f',decimals(maximum()))
                           .arg(toolTip()));
    //setValidator(new QDoubleValidator(minimum(), maximum(), 1000));
    setValidator(new QDoubleValidator());// minimum(), maximum(), 1000));
}


void ValueSlider::
        sliderMoved(int v)
{
    value_ = toReal(v);
    emit valueChanged(value_);
}


void ValueSlider::
        sliderPressed()
{
    slider_is_pressed_ = true;
}


void ValueSlider::
        sliderReleased()
{
    slider_is_pressed_ = false;
    slider_->setValue (toInt(value_));
}


void ValueSlider::
        updateLineEdit()
{
    lineEdit()->setText(valueAsString());
}


void ValueSlider::
        editingFinished()
{
    QString text = currentText();
    if (text == valueAsString(true))
        return;

    bool ok = false;
    qreal v = text.toDouble(&ok);

    if (!ok)
        v = value();

    setValue( v );
}


void ValueSlider::
        returnPressed()
{
    setValue(value_);
}


QString ValueSlider::
        valueAsString(bool forceDisableUnits) const
{
    return QString("%1%2").arg(value_,0,'f',decimals(value_)).arg (unit_.isEmpty () || hasFocus () || forceDisableUnits?"":" " + unit_);
}


int ValueSlider::
        decimals(qreal r) const
{
    int d = decimals();

    r = fabs(r);

    if (r>0)
    {
        while(r<0.099)
        {
            r*=10;
            d++;
        }
        while(r>=1.1 && d>0)
        {
            r/=10;
            d--;
        }
    }

    return d;

}


qreal ValueSlider::
        toReal(int v) const
{
    qreal f = qreal(v)/resolution_;

    switch(value_translation_)
    {
    case Quadratic:
        return f*f;

    case LogaritmicZeroMin:
        if (v <= slider_->minimum ())
            return 0;

        // fall through to Logaritmic
    case Logaritmic:
        return exp(f);

    case Linear:
    default:
        return f;
    }
}


int ValueSlider::
        toInt(qreal v) const
{
    switch(value_translation_)
    {
    case Quadratic:
        if (v < 0)
            return slider_->minimum ();

        v = sqrt(v);
        break;

    case Logaritmic:
    case LogaritmicZeroMin:
        if (v < 0)
            return slider_->minimum ();

        v = log(v);
        break;

    case Linear:
    default:
        break;
    }

    return v*resolution_ + 0.5;
}


} // namespace Support
} // namespace Tools
