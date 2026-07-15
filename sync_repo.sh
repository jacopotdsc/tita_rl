
#!/bin/bash

TARGET="${1:-none}"
FILE_ARG="${2:-}" 

if [ "$TARGET" = "pc2070" ]; then
  REMOTE="jacopo@192.168.40.45" # Ethernet
  REMOTE_BASE="/home/jacopo"
  ENV_NAME="mjpl"
  PYTHON_VERSION="3.11"

elif [ "$TARGET" = "pc5090" ]; then
  REMOTE="jacopo@192.168.40.157"
  REMOTE_BASE="/home/jacopo"
  ENV_NAME="mjpl"
  PYTHON_VERSION="3.11"

elif [ "$TARGET" = "mio" ]; then
  REMOTE="ubuntu@10.90.234.27"
  REMOTE_BASE="/home/ubuntu"
  ENV_NAME="jax_env"
  PYTHON_VERSION="3.13"

else
  echo "Unknown target: $TARGET"
  echo "Available targets: pc2070, pc5090, mio"
  echo "Check if REMOTE and REMOTE_BASE are set correctly in the script."
  echo "REMOTE: $REMOTE"
  echo "REMOTE_BASE: $REMOTE_BASE"
  exit 1
fi

EXCLUDES=(
  --exclude 'jax_cache/'
  --exclude '.jax_cache/'
  --exclude '__pycache__/'
  --exclude '*.pyc'
  --exclude '.pytest_cache/'
  --exclude '.cache/'
  --exclude 'warp_cache/'
)

if [ -n "$FILE_ARG" ]; then
  LOCAL_BASE="$HOME/Desktop/repo_rl/tita_rl/test/mpx/mpx/examples"
  REMOTE_BASE_EXAMPLES="$REMOTE_BASE/Desktop/repo_rl/tita_rl/test/mpx/mpx/examples"
 
  SRC="$LOCAL_BASE/$FILE_ARG"
  # Keep sub-path on the remote side (e.g. 'configs/foo.py' -> examples/configs/)
  SUBDIR=$(dirname "$FILE_ARG")
  if [ "$SUBDIR" = "." ]; then
    DST="$REMOTE:$REMOTE_BASE_EXAMPLES/"
  else
    DST="$REMOTE:$REMOTE_BASE_EXAMPLES/$SUBDIR/"
  fi
 
  if [ ! -e "$SRC" ]; then
    echo "Source does not exist: $SRC"
    exit 1
  fi
 
  read -p "Copy '$FILE_ARG' to $TARGET ($REMOTE)? [y/N]: " confirm
  if [[ "$confirm" != "y" && "$confirm" != "Y" ]]; then
    echo "Aborted."
    exit 0
  fi
 
  rsync -avz --copy-links --checksum "${EXCLUDES[@]}" "$SRC" "$DST"
  echo "Done (copied only '$FILE_ARG')."
  exit 0
fi


read -p "Do you want to copy to $TARGET ($REMOTE)? [y/N]: " confirm
if [[ "$confirm" != "y" && "$confirm" != "Y" ]]; then
  echo "Aborted."
  exit 0
fi

echo "Sync repo_rl..."
rsync -avz --copy-links --checksum --delete \
  "${EXCLUDES[@]}" \
  /home/jacopo/Desktop/repo_rl/tita_rl \
  --exclude '/home/jacopo/Desktop/repo_rl/tita_rl/test/mpx/mpx/examples/jax_cache/' \
  $REMOTE:$REMOTE_BASE/Desktop/repo_rl/

echo "Sync mujoco_playground..."
rsync -avz --copy-links --checksum --delete \
  "${EXCLUDES[@]}" \
  /home/jacopo/miniconda3/envs/mjpl/lib/python3.11/site-packages/mujoco_playground/ \
  $REMOTE:$REMOTE_BASE/miniconda3/envs/$ENV_NAME/lib/python$PYTHON_VERSION/site-packages/mujoco_playground/

echo "Done."