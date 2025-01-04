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
  make -j6
  cd $AXOMAE_ROOT
  ln -sf $RELWITHDEBINFO/$COMPILE_COMMANDS_FILE ./ 
}


build_debug(){
  check_current_dir
  cd $DEBUG
  make -j6
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
