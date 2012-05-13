#ifndef COMMANDEDIT_H
#define COMMANDEDIT_H

#include <QLineEdit>

namespace Tools {
namespace Support {

class CommandEdit : public QLineEdit
{
    Q_OBJECT
public:
    explicit CommandEdit(QWidget *parent = 0);

protected:
    virtual void keyPressEvent ( QKeyEvent * event );

private slots:
    void returnPressedSlot();

private:
    std::vector<QString> commands;
    unsigned command_index;
};

} // namespace Support
} // namespace Tools

#endif // COMMANDEDIT_H
