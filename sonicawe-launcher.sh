#!/bin/bash
set -e

sonicawe-cuda $* || sonicawe-cpu $*
