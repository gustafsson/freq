#include <QtGui/QApplication>
#include <iostream>
#include <stdio.h>

#include "mainwindow.h"
#include "displaywidget.h"
#include "transform.h"

using namespace std;

static const char _sawe_usage_string[] =
        "sawe [--version] [--samples_per_chunk[=#n]] [--scales_per_octave[=#n]]\n"
        "           [--help] [FILENAME]\n";

static unsigned _samples_per_chunk = 1<<13;
static unsigned _scales_per_octave = 40;
static const char* _soundfile = "input.wav";

static int prefixcmp(const char *a, const char *prefix) {
    for(;a && prefix;a++,prefix++) {
        if (*a < *prefix) return -1;
        if (*a > *prefix) return 1;
    }
    return 0!=prefix;
}

static int handle_options(char ***argv, int *argc)
{
    int handled = 0;
    bool sawe_exit=false;

    while (*argc > 0) {
        const char *cmd = (*argv)[0];
        if (cmd[0] != '-')
            break;

        if (!strcmp(cmd, "--help")) {
            printf("%s", _sawe_usage_string);
            sawe_exit = true;
        } else if (!strcmp(cmd, "--version")) {
            printf("TODO: print version info\n");
            sawe_exit = true;
        } else if (!prefixcmp(cmd, "--samples_per_chunk")) {
            cmd += strlen("--samples_per_chunk");
            if (*cmd == '=')
                _samples_per_chunk = atoi(cmd+1);
            else {
                printf("default samples_per_chunk=%d\n", _samples_per_chunk);
                sawe_exit = true;
            }
        } else if (!prefixcmp(cmd, "--scales_per_octave")) {
            cmd += strlen("--scales_per_octave");
            if (*cmd == '=')
                _scales_per_octave = atoi(cmd+1);
            else {
                printf("default scales_per_octave=%d\n", _scales_per_octave);
                sawe_exit = true;
            }
        } else {
            fprintf(stderr, "Unknown option: %s\n", cmd);
            printf("%s", _sawe_usage_string);
            exit(0);
        }

        (*argv)++;
        (*argc)--;
        handled++;
    }

    if (sawe_exit)
        exit(0);

    return handled;
}

int main(int argc, char *argv[])
{
    QApplication a(argc, argv);
    MainWindow w;

    while (argc) {
        handle_options(&argv, &argc);

        if (argc) {
            _soundfile = argv[0];
            argv++;
            argc--;
        }
    }

    boost::shared_ptr<Waveform> wf( new Waveform( _soundfile ) );
    boost::shared_ptr<Transform> wt( new Transform(wf, _scales_per_octave, _samples_per_chunk ) );
/*    boost::shared_ptr<DisplayWidget> dw( new DisplayWidget( wt ) );

    w.setCentralWidget( dw.get() );
    dw->show();
    w.show();*/

   return a.exec();
}
