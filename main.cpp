#include <QtGui/QApplication>
#include <iostream>
#include <stdio.h>

#include "mainwindow.h"
#include "displaywidget.h"
#include "transform.h"

using namespace std;

static const char _sawe_version_string[] =
        "sawe version 0.0.2\n";

static const char _sawe_usage_string[] =
        "sawe [--samples_per_chunk=#n] [--scales_per_octave=#n]\n"
        "           [--samples_per_block=#n] [--scales_per_block=#n]\n"
        "           [--channel=#n] FILENAME\n"
        "sawe [--samples_per_chunk] [--scales_per_octave]\n"
        "           [--samples_per_block] [--scales_per_block]\n"
        "           [--channel]\n"
        "sawe [--help] \n"
        "sawe [--version] \n";

static unsigned _channel=0;
static unsigned _samples_per_chunk = 1<<13;
static unsigned _scales_per_octave = 40;
static float _wavelet_std_t = 0.03;
static unsigned _samples_per_block = 1<<9;
static unsigned _scales_per_block = 1<<8;
static const char* _soundfile = 0;
static bool _sawe_exit=false;

static int prefixcmp(const char *a, const char *prefix) {
    for(;*a && *prefix;a++,prefix++) {
        if (*a < *prefix) return -1;
        if (*a > *prefix) return 1;
    }
    return 0!=*prefix;
}


void atoval(const char *cmd, float& val) {
    val = atof(cmd);
}
void atoval(const char *cmd, unsigned& val) {
    val = atoi(cmd);
}

#define readarg(cmd, name) tryreadarg(cmd, "--"#name, #name, _##name)

template<typename Type>
bool tryreadarg(const char **cmd, const char* prefix, const char* name, Type &val) {
    if (prefixcmp(*cmd, prefix))
        return 0;
    *cmd += strlen(prefix);
    if (**cmd == '=')
        atoval(*cmd+1, val);
    else {
        cout << "default " << name << "=" << val << endl;
        _sawe_exit = true;
    }
    return 1;
}

static int handle_options(char ***argv, int *argc)
{
    int handled = 0;

    while (*argc > 0) {
        const char *cmd = (*argv)[0];
        if (cmd[0] != '-')
            break;

        if (!strcmp(cmd, "--help")) {
            printf("%s", _sawe_usage_string);
            _sawe_exit = true;
        } else if (!strcmp(cmd, "--version")) {
            printf("%s", _sawe_version_string);
            _sawe_exit = true;
        }
        else if (readarg(&cmd, samples_per_chunk));
        else if (readarg(&cmd, scales_per_octave));
        else if (readarg(&cmd, wavelet_std_t));
        else if (readarg(&cmd, samples_per_block));
        else if (readarg(&cmd, scales_per_block));
        else {
            fprintf(stderr, "Unknown option: %s\n", cmd);
            printf("%s", _sawe_usage_string);
            exit(1);
        }

        (*argv)++;
        (*argc)--;
        handled++;
    }

    if (_sawe_exit)
        exit(0);

    return handled;
}

int main(int argc, char *argv[])
{
    QApplication a(argc, argv);
    MainWindow w;

    // skip application filename
    argv++;
    argc--;

    while (argc) {
        handle_options(&argv, &argc);

        if (argc) {
            _soundfile = argv[0];
            argv++;
            argc--;
        }
    }

    if (0 == _soundfile) {
        printf("Missing audio file\n\n%s", _sawe_usage_string);
        exit(1);
    }

    try
    {
        boost::shared_ptr<Waveform> wf( new Waveform( _soundfile ) );
        boost::shared_ptr<Transform> wt( new Transform(wf, _channel, _samples_per_chunk, _scales_per_octave, _wavelet_std_t ) );
        boost::shared_ptr<Spectrogram> sg( new Spectrogram(wt, _samples_per_block, _scales_per_block  ) );
        boost::shared_ptr<DisplayWidget> dw( new DisplayWidget( sg ) );

        w.setCentralWidget( dw.get() );
        dw->show();
        w.show();

       return a.exec();
   } catch ( const std::exception& x ) {
       cout << "Caught exception: " << x.what() << endl;
   }
}
