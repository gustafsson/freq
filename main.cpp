#include <QtGui/QApplication>
#include "mainwindow.h"
#include "displaywidget.h"
#include "transform.h"
#include <iostream>
using namespace std;

const char sawe_usage_string[] =
        "sawe [--version] [--samples_per_chunk[=#n]] [--scales_per_octave[=#n]]\n"
        "           [--help] [FILENAME]\n";

struct sawe_options {
   unsigned samples_per_chunk = 1<<13;
   unsigned scales_per_octave = 40;
   const char* soundfile = "input.wav";
} gSawe_options;

static int prefixcmp(const char *a, const char *prefix) {
    for(;a && prefix;a++,prefix++) {
        if (*a < *prefix) return -1;
        if (*a > *prefix) return 1;
    }
    return 0!=prefix;
}

static int handle_options(const char ***argv, int *argc)
{
    int handled = 0;
    bool sawe_exit=false;

    while (*argc > 0) {
        const char *cmd = (*argv)[0];
        if (cmd[0] != '-')
            break;

        if (!strcmp(cmd, "--help")) {
            printf(sawe_usage_string);
            sawe_exit = true;
        } else if (!strcmp(cmd, "--version")) {
            printf("TODO: print version info\n");
            sawe_exit = true;
        } else if (!prefixcmp(cmd, "--samples_per_chunk")) {
            cmd += strlen("--samples_per_chunk");
            if (*cmd == '=')
                gSawe_options.samples_per_chunk = atoi(cmd+1);
            else {
                printf("default samples_per_chunk=%d\n", gSawe_options.samples_per_chunk);
                sawe_exit = true;
            }
        } else if (!prefixcmp(cmd, "--scales_per_octave")) {
            cmd += strlen("--scales_per_octave");
            if (*cmd == '=')
                gSawe_options.scales_per_octave = atoi(cmd+1);
            else {
                printf("default scales_per_octave=%d\n", gSawe_options.scales_per_octave);
                sawe_exit = true;
            }
        } else {
            fprintf(stderr, "Unknown option: %s\n", cmd);
            printf(sawe_usage_string);
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

        gSawe_options.soundfile = argv[0];
        argv++;
        argc--;
    }

    boost::shared_ptr<Waveform> wf( new Waveform( gSawe_options.soundfile ) );
    boost::shared_ptr<Transform> wt( new Transform(wf, gSawe_options.scales_per_octave, gSawe_options.samples_per_chunk ) );
    boost::shared_ptr<DisplayWidget> dw( new DisplayWidget( wt ) );

    w.setCentralWidget( dw.get() );
    dw->show();
    w.show();

   return a.exec();
}
