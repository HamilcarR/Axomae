#!/bin/sh


# refactor or integrate fully to cmake
PROJECT_DIR=$1
shader_path=$PROJECT_DIR/sources/gpu/opengl/shader
echo "Moving to $shader_path" 
cd $shader_path 

echo "Executing stringify.sh ..."
echo "Generating vertex shaders strings"
for f in glsl/*.vert; do
    xxd -i glsl/"${f##*/}" >> utils/temp_vertex.h
done

echo "Generating fragment shaders strings"
for f in glsl/*.frag; do
    echo $f;
    xxd -i glsl/"${f##*/}" >> utils/temp_fragment.h
done



if [ -f utils/vertex.h ]; then
    rm utils/vertex.h
fi

if [ -f utils/fragment.h ]; then
    rm utils/fragment.h
fi

mv utils/temp_vertex.h utils/vertex.h
mv utils/temp_fragment.h utils/fragment.h

echo "Shaders generation done..."
