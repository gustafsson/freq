#!/bin/bash
set -e

git checkout @{"`cd ../sonicawe;git log -1 --format=%ci`"}
