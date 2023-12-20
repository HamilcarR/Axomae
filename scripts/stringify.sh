#!/bin/sh

PROJECT_DIR=$1
echo $PROJECT_DIR
cd $PROJECT_DIR/sources/gpu/shader/shaders

echo "Executing stringify.sh ..."
echo "Generating vertex shaders strings"
for f in glsl/*.vert; do
    xxd -i glsl/"${f##*/}" >>includes/temp_vertex.h
done

echo "Generating fragment shaders strings"
for f in glsl/*.frag; do
    xxd -i glsl/"${f##*/}" >>includes/temp_fragment.h
done

if [ -f includes/vertex.h ]; then
    rm includes/vertex.h
fi

if [ -f includes/fragment.h ]; then
    rm includes/fragment.h
fi
mv includes/temp_vertex.h includes/vertex.h
mv includes/temp_fragment.h includes/fragment.h
