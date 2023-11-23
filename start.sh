#!/usr/bin/env bash

# params
system_version="windows";
verbose=true;

port=9080

# parse options
while true; do
  [ -z "${1:-}" ] && break;  # break if there are no arguments
  case "$1" in
    --*) name=$(echo "$1" | sed s/^--// | sed s/-/_/g);
      eval '[ -z "${'"$name"'+xxx}" ]' && echo "$0: invalid option $1" 1>&2 && exit 1;
      old_value="(eval echo \\$$name)";
      if [ "${old_value}" == "true" ] || [ "${old_value}" == "false" ]; then
        was_bool=true;
      else
        was_bool=false;
      fi

      # Set the variable to the right value-- the escaped quotes make it work if
      # the option had spaces, like --cmd "queue.pl -sync y"
      eval "${name}=\"$2\"";

      # Check that Boolean-valued arguments are really Boolean.
      if $was_bool && [[ "$2" != "true" && "$2" != "false" ]]; then
        echo "$0: expected \"true\" or \"false\": $1 $2" 1>&2
        exit 1;
      fi
      shift 2;
      ;;

    *) break;
  esac
done


$verbose && echo "system_version: ${system_version}"


if [ $system_version == "windows" ]; then
  alias python3='C:/Users/tianx/PycharmProjects/virtualenv/CallbotNLPServer/Scripts/python.exe'
elif [ $system_version == "centos" ]; then
  # conda activate CallbotNLPServer
  alias python3='/usr/local/miniconda3/envs/CallbotNLPServer/bin/python3'
elif [ $system_version == "ubuntu" ]; then
  alias python3='/usr/local/miniconda3/envs/CallbotNLPServer/bin/python3'
fi


cd "$(pwd)/server/callbot_nlp_server/" || exit 1;

nohup python3 run_callbot_nlp_server.py --port "${port}" > nohup_run.out &

sleep 10

tail -f log/server.log
