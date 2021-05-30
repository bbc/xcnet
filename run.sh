#!/bin/sh

help() {
    echo
    echo "\
    Options:
      -h; Show help.
      -a <action>; Select action (data | train | test).
      -d <path>; Path to dataset.
      -e <name>; Experiment name (default: xcnet) -> output_path: $(pwd)/xcnet.
      -c <path>; Path to config file (default: $(pwd)/src/config/xcnet.py).
      -g <number>; Cuda device number (default: 0).
      -m <number>; Memory shm-size (G) (default: 8).
    " | column -t -s ";"
}

error() { help; exit 1; }

# Defaults
DEVICE=0
MEMORY=8G
NAME=xcnet
CONFIG=$(pwd)/src/config/xcnet.py

while getopts "ha:d:e:c:g:m:" OPTIONS; do
    case $OPTIONS in
        h) help; exit 1;;
        a) ACTION=${OPTARG} ;;
        d) DATA=${OPTARG} ;;
        e) NAME=${OPTARG} ;;
        c) CONFIG=${OPTARG} ;;
        g) DEVICE=${OPTARG} ;;
        m) MEMORY=${OPTARG}G ;;
    esac
done

[ -z $DATA ] && { echo "Error: Provide path to database."; error; }

case $ACTION in
    data)
        nvidia-docker run -it -e NVIDIA_VISIBLE_DEVICES=$DEVICE \
        --mount "type=bind,src=$(pwd)/src,dst=/app/src" \
        --mount "type=bind,src=$DATA,dst=/app/dataset" \
        --user $(id -u):$(id -g) xcnet \
        python src/data/process_data.py -d /app/dataset
        nvidia-docker run -it -e NVIDIA_VISIBLE_DEVICES=$DEVICE \
        --mount "type=bind,src=$(pwd)/src,dst=/app/src" \
        --mount "type=bind,src=$DATA,dst=/app/dataset" \
        --user $(id -u):$(id -g) xcnet \
        python src/data/create_database.py -d /app/dataset -g "cuda:$DEVICE" ;;

    train)
        mkdir -p experiments/$NAME; cp $CONFIG experiments/$NAME/config.py
        nvidia-docker run -it -e NVIDIA_VISIBLE_DEVICES=$DEVICE \
        --mount "type=bind,src=$(pwd)/src,dst=/app/src" \
        --mount "type=bind,src=$DATA,dst=/app/dataset" \
        --mount "type=bind,src=$(pwd)/experiments/$NAME,dst=/app/experiments/$NAME" \
        --mount "type=bind,src=$(pwd)/.cache,dst=/home/user/.cache/" \
        --user $(id -u):$(id -g) --shm-size $MEMORY xcnet \
        python src/train.py -d /app/dataset -c /app/experiments/$NAME/config.py -o /app/experiments/$NAME -g "cuda:0" ;;

    test)
        cp $CONFIG experiments/$NAME/config.py
        nvidia-docker run -it -e NVIDIA_VISIBLE_DEVICES=$DEVICE \
        --mount "type=bind,src=$(pwd)/src,dst=/app/src" \
        --mount "type=bind,src=$DATA,dst=/app/dataset" \
        --mount "type=bind,src=$(pwd)/experiments/$NAME,dst=/app/experiments/$NAME" \
        --mount "type=bind,src=$(pwd)/.cache,dst=/home/user/.cache/" \
        --user $(id -u):$(id -g) --shm-size $MEMORY xcnet \
        python src/test.py -d /app/dataset -c /app/experiments/$NAME/config.py -o /app/experiments/$NAME -g "cuda:0" ;;
    *) echo "Error: Select a valid action."; error ;;
esac