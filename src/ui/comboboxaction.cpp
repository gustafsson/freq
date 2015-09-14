#include "comboboxaction.h"
#include <QAction>

namespace Ui
{

ComboBoxAction::
        ComboBoxAction(QWidget * parent)
            :   QToolButton(parent),
                _decheckable(true)
{
    connect( this, SIGNAL(triggered(QAction *)), this, SLOT(checkAction(QAction *)));
    this->setContextMenuPolicy( Qt::ActionsContextMenu );
}

void ComboBoxAction::
        addActionItem( QAction* a )
{
    addAction( a );
    if (0 == defaultAction())
        setDefaultAction(a);
}

void ComboBoxAction::
        setCheckedAction( QAction* a )
{
    if (!actions().contains(a))
        addActionItem(a);
    if (a != defaultAction())
    {
        defaultAction()->setChecked(false);
    }

    setDefaultAction(a);
    a->setChecked(true);
    setChecked(true);
}

void ComboBoxAction::
        decheckable( bool a )
{
    _decheckable = a;
    if (false == _decheckable)
        setChecked( true );
}

void ComboBoxAction::
        checkAction( QAction* a )
{
    if (a->isChecked())
    {
        QList<QAction*> l = actions();
        for (QList<QAction*>::iterator i = l.begin(); i!=l.end(); i++)
            if (*i != a)
                (*i)->setChecked( false );
    }

    if (false == _decheckable)
        a->setChecked( true );

    setDefaultAction( a );
}

} // namespace Ui
