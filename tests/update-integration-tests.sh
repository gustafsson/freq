#!/bin/bash
set -e

(
    cd ..
    git submodule update --init tests/integration
)
