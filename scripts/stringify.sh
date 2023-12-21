#!/bin/sh

PROJECT_DIR=$1
echo $PROJECT_DIR
cd $PROJECT_DIR/sources/gpu/opengl/shader

echo "Executing stringify.sh ..."
echo "Generating vertex shaders strings"
for f in glsl/*.vert; do
    xxd -i glsl/"${f##*/}" >>utils/temp_vertex.h
done

echo "Generating fragment shaders strings"
for f in glsl/*.frag; do
    xxd -i glsl/"${f##*/}" >>utils/temp_fragment.h
done

if [ -f utils/vertex.h ]; then
    rm utils/vertex.h
fi

if [ -f utils/fragment.h ]; then
    rm utils/fragment.h
fi
mv utils/temp_vertex.h utils/vertex.h
mv utils/temp_fragment.h utils/fragment.h
