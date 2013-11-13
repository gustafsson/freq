#ifndef UI_COMBOBOXACTION_H
#define UI_COMBOBOXACTION_H

#include <QToolButton>

namespace Ui
{

class ComboBoxAction: public QToolButton {
    Q_OBJECT
public:
    ComboBoxAction(QWidget * parent=0);
    void addActionItem( QAction* a );
    void setCheckedAction( QAction* a );
    void decheckable(bool);

private slots:
    virtual void checkAction( QAction* a );

private:
    bool _decheckable;
};

} // namespace Ui

#endif // UI_COMBOBOXACTION_H
