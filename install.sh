#!/usr/bin/env bash

# bash install.sh --stage 1 --stop_stage 1 --system_version centos
# bash install.sh --stage 3 --stop_stage 4 --system_version centos
# bash install.sh --stage 5 --stop_stage 5 --system_version centos


python_version=3.6.5
system_version=centos

verbose=true;
stage=-1
stop_stage=2

work_dir="$(pwd)"


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


data_dir="$(pwd)/data"

mkdir -p "${data_dir}"

if [ $system_version == "centos" ]; then
  yum install -y git;
elif [ $system_version == "ubuntu" ]; then
  apt-get install -y git

fi


if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
  $verbose && echo "stage 1: install python"
  cd "${work_dir}" || exit 1;

  sh ./script/install_python.sh --python_version "${python_version}" --system_version "${system_version}"
fi


if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
  $verbose && echo "stage 2: create virtualenv"
  /usr/local/python-${python_version}/bin/pip3 install virtualenv
  mkdir -p /data/local/bin
  cd /data/local/bin || exit 1;
  # source /data/local/bin/CallbotNLPServer/bin/activate
  /usr/local/python-${python_version}/bin/virtualenv CallbotNLPServer

fi


if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
  $verbose && echo "stage 3: download ltp_data_v3.4.0.zip"
  cd "${data_dir}" || exit 1;
  wget http://model.scir.yunfutech.com/model/ltp_data_v3.4.0.zip
  unzip ltp_data_v3.4.0.zip
  rm -rf ltp_data_v3.4.0.zip

fi


if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
  $verbose && echo "stage 3: download unidic-3.1.0.zip"
  cd "${data_dir}" || exit 1;
  wget https://cotonoha-dic.s3-ap-northeast-1.amazonaws.com/unidic-3.1.0.zip
  unzip unidic-3.1.0.zip
  rm -rf unidic-3.1.0.zip

fi

if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
  $verbose && echo "stage 5: download nltk_data"
  cd "${data_dir}" || exit 1;

  # http://www.nltk.org/nltk_data/
  wget https://huggingface.co/spaces/qgyd2021/nlp_tools/resolve/main/data/nltk_data.zip
  unzip nltk_data.zip
  rm -rf nltk_data.zip

fi
