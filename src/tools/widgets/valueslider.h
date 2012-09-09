#ifndef TOOLS_SUPPORT_VALUESLIDER_H
#define TOOLS_SUPPORT_VALUESLIDER_H

#include <QComboBox>

namespace Tools {
namespace Widgets {

class ValueSlider : public QComboBox
{
    Q_OBJECT

    Q_PROPERTY(qreal min READ min WRITE setMin)
    Q_PROPERTY(qreal max READ max WRITE setMax)
    Q_PROPERTY(qreal value READ value WRITE setValue)
    Q_PROPERTY(qreal decimals READ decimals WRITE setDecimals)

public:
    explicit ValueSlider(QWidget *parent = 0);
    ~ValueSlider();

    // QComboBox
    void showPopup();
    void hidePopup();
    void focusOutEvent ( QFocusEvent * e );
    bool eventFilter(QObject *, QEvent *);

    qreal min() const;
    void setMin(qreal);
    qreal max() const;
    void setMax(qreal);
    qreal value() const;
    void setValue(qreal v);

    /// Number of decimal digits.
    int decimals() const;
    void setDecimals(int);

signals:
    void valueChanged(qreal value);
    void rangeChanged(qreal min, qreal max);
    void sliderMoved(qreal value);

private slots:
    void valueChanged(int);
    void rangeChanged(int,int);
    void sliderMoved(int);
    void editingFinished();

private:    
    QDialog* popup_;
    QSlider* slider_;
    int resolution_;

    qreal toReal(int) const;
    int toInt(qreal) const;
};

} // namespace Support
} // namespace Tools

#endif // TOOLS_SUPPORT_VALUESLIDER_H
