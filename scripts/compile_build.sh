#! /bin/bash 

BUILD_TYPE=$1

AXOMAE_ROOT=$(pwd)
RELWITHDEBINFO="$AXOMAE_ROOT/../build_RelWithDebInfo"
DEBUG="$AXOMAE_ROOT/../build_Debug"
COMPILE_COMMANDS_FILE="compile_commands.json"


show_info(){
  echo "" 
  echo "Usage: ./script/compile_build.sh [OPTIONS]"
  echo ""
  echo "Utility script to build with a premade configuration. Execute in root project folder."
  echo ""
  echo "Options:"
  echo "  reldb          Build Release With Debug Info."
  echo "  debug          Build Debug."
  echo "  help           Show this help message and exit." 
}


markers=(".git/" ".clang-tidy" ".clang-format" "LICENSE" "README.md" ".gitmodules" ".gitignore" ".gitattributes" )

check_current_dir(){
  for marker in "${markers[@]}"; do 
    if [ ! -e $marker ]; then 
      echo "$marker"
      printf "Not in project root.\nExecute this script from the project root dir.\n" ; 
      exit 1
    fi
    done
}


build_reldb(){
  check_current_dir
  cd $RELWITHDEBINFO
  CMAKE_BUILD_COMMAND_RELDB="-DAXOMAE_BUILD_TESTS:BOOL=ON -DAXOMAE_USE_CUDA:BOOL=ON -DCMAKE_CUDA_ARCHITECTURES:STRING=75 -DAXOMAE_FROMSOURCE_QT_BUILD:BOOL=ON -DASSIMP_BUILD_ALL_EXPORTERS_BY_DEFAULT:BOOL=OFF -DIMATH_INSTALL:BOOL=OFF -DIMATH_INSTALL_PKG_CONFIG:BOOL=OFF -DIMATH_INSTALL_SYM_LINK:BOOL=OFF -DOPENEXR_BUILD_EXAMPLES:BOOL=OFF -DOPENEXR_BUILD_TOOLS:BOOL=OFF -DOPENEXR_INSTALL:BOOL=OFF -DOPENEXR_INSTALL_PKG_CONFIG:BOOL=OFF -DAXOMAE_USE_SCCACHE:BOOL=ON"
  cmake -S $AXOMAE_ROOT -G "Unix Makefiles" $CMAKE_BUILD_COMMAND_RELDB -DCMAKE_BUILD_TYPE:STRING=RelWithDebugInfo -B . 
  make -j10
  cd $AXOMAE_ROOT
  ln -sf $RELWITHDEBINFO/$COMPILE_COMMANDS_FILE ./ 
}


build_debug(){
  check_current_dir
  cd $DEBUG
  CMAKE_BUILD_COMMAND_RELDB="-DAXOMAE_BUILD_TESTS:BOOL=ON -DAXOMAE_USE_CUDA:BOOL=OFF -DAXOMAE_FROMSOURCE_QT_BUILD:BOOL=ON -DASSIMP_BUILD_ALL_EXPORTERS_BY_DEFAULT:BOOL=OFF -DIMATH_INSTALL:BOOL=OFF -DIMATH_INSTALL_PKG_CONFIG:BOOL=OFF -DIMATH_INSTALL_SYM_LINK:BOOL=OFF -DOPENEXR_BUILD_EXAMPLES:BOOL=OFF -DOPENEXR_BUILD_TOOLS:BOOL=OFF -DOPENEXR_INSTALL:BOOL=OFF -DOPENEXR_INSTALL_PKG_CONFIG:BOOL=OFF -DAXOMAE_USE_SCCACHE:BOOL=ON"
  cmake -S $AXOMAE_ROOT -G "Unix Makefiles" $CMAKE_BUILD_COMMAND_RELDB -DCMAKE_BUILD_TYPE:STRING=Debug -B . 
  make -j10
  cd $AXOMAE_ROOT
  ln -sf $DEBUG/$COMPILE_COMMANDS_FILE ./ 
}



if [[ "$BUILD_TYPE" == "reldb" ]]; then 
  build_reldb
elif [[ "$BUILD_TYPE" == "debug" ]]; then 
  build_debug
else
  show_info
fi
