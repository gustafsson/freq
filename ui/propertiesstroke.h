#ifndef PROPERTIESSTROKE_H
#define PROPERTIESSTROKE_H

#include <QWidget>

namespace Ui {

namespace Ui {
    class PropertiesStroke;
}

class PropertiesStroke : public QWidget {
    Q_OBJECT
public:
    PropertiesStroke(QWidget *parent = 0);
    ~PropertiesStroke();

protected:
    void changeEvent(QEvent *e);

private:
    Ui::PropertiesStroke *ui;
};


} // namespace Ui
#endif // PROPERTIESSTROKE_H
