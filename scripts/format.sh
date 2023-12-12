#!/bin/sh

if [ -z "$1" ]; then
    clang-format -i -style=file includes/*.h*
    clang-format -i -style=file sources/*.cpp
    clang-format -i -style=file shaders/*
else
    clang-format -i -style=file "$1"
fi
