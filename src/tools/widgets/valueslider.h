#ifndef TOOLS_SUPPORT_VALUESLIDER_H
#define TOOLS_SUPPORT_VALUESLIDER_H

#include <QComboBox>

namespace Tools {
namespace Widgets {

class ValueSlider : public QComboBox
{
    Q_OBJECT

    Q_PROPERTY(qreal minimum READ minimum WRITE setMinimum)
    Q_PROPERTY(qreal maximum READ maximum WRITE setMaximum)
    Q_PROPERTY(qreal value READ value WRITE setValue)
    Q_PROPERTY(qreal decimals READ decimals WRITE setDecimals)

public:
    enum ValueTranslation
    {
        Linear,
        Quadratic,
        Logaritmic,
        LogaritmicZeroMin
    };

    explicit ValueSlider(QWidget *parent = 0);
    ~ValueSlider();

    // QComboBox
    void showPopup();
    void hidePopup();
    void focusInEvent ( QFocusEvent * e );
    void focusOutEvent ( QFocusEvent * e );
    bool eventFilter(QObject *, QEvent *);

    qreal minimum() const;
    void setMinimum(qreal);
    qreal maximum() const;
    void setMaximum(qreal);
    qreal value() const;
    void setValue(qreal v);
    void setRange(qreal min, qreal max, ValueTranslation translation=Linear);

    QString unit() const;
    void setUnit(QString);

    Qt::Orientation orientation();
    void setOrientation( Qt::Orientation orientation );

    QString toolTip();
    void setToolTip( QString str );

    void triggerAction ( QAbstractSlider::SliderAction action );

    ValueTranslation valueTranslation();
    void setValueTranslation(ValueTranslation translation);

    /// Number of decimal digits to print.
    int decimals() const;
    void setDecimals(int);

    void setSliderSize(int);

    void updateLineEditOnValueChanged(bool);

signals:
    void valueChanged(qreal value);
    void rangeChanged(qreal minimum, qreal maximum);

private slots:
    void valueChanged(int);
    void rangeChanged(int,int);
    void sliderMoved(int);
    void editingFinished();
    void returnPressed();

    void sliderPressed();
    void sliderReleased();
    void updateLineEdit();

private:
    QDialog* popup_;
    QSlider* slider_;
    int resolution_;
    int decimals_;
    ValueTranslation value_translation_;
    qreal value_;
    bool slider_is_pressed_;
    QString unit_;
    bool logaritmic_zero_;

    QString valueAsString(bool forceDisableUnits=false) const;
    int decimals(qreal) const;
    qreal toReal(int) const;
    int toInt(qreal) const;
};

} // namespace Support
} // namespace Tools

#endif // TOOLS_SUPPORT_VALUESLIDER_H
