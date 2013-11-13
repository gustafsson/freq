#!/bin/bash
set -e
mypath=$(dirname "$0")
"$mypath/sonicawe-cuda" $* || "$mypath/sonicawe-cpu" $*
