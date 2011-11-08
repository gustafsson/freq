#!/bin/bash
set -e

sonicawe-reader-cuda $* || sonicawe-reader $*
