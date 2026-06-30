
#!/usr/bin/env bash
mkdir -p ros_ws/src

if [ ! -d ros_ws/src/motion_capture_tracking/.git ]; then
  echo "[Pixi activation] Cloning motion_capture_tracking..."
  git clone --recurse-submodules https://github.com/learnsyslab/motion_capture_tracking ros_ws/src/motion_capture_tracking
fi

if [ ! -f ros_ws/install/setup.sh ]; then
  echo "[Pixi activation] Running colcon build..."
  (cd ros_ws && colcon build --cmake-args -DCMAKE_POLICY_VERSION_MINIMUM=3.5)
fi

# Colcon/ament generated setup scripts read optional environment variables directly. Pixi can run
# activation scripts with nounset enabled, so source the generated setup with nounset temporarily
# disabled and then restore the caller's shell flags.
case $- in
  *u*)
    _setup_mocap_restore_nounset=1
    set +u
    ;;
  *)
    _setup_mocap_restore_nounset=0
    ;;
esac
. ./ros_ws/install/setup.sh
if [ "$_setup_mocap_restore_nounset" = 1 ]; then
  set -u
fi
unset _setup_mocap_restore_nounset
