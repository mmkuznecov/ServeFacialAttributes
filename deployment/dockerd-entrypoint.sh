#!/bin/bash
set -e

# Check for 'serve' command and start TorchServe
if [[ "$1" = "serve" ]]; then
    bash start_torchserve.sh
else
    eval "$@"
fi

# Keep the container running (This is useful if you want to get into the container and execute commands)
tail -f /dev/null
