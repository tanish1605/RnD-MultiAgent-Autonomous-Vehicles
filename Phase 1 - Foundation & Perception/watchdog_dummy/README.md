# watchdog_dummy

A minimal ROS 2 Humble Python node for testing a watchdog that receives
newline-delimited JSON heartbeats over an AF_UNIX socket.

## Build

Copy this package into `~/ros2_ws/src/watchdog_dummy`, then run:

```bash
cd ~/ros2_ws
colcon build --packages-select watchdog_dummy
source install/setup.bash
```

## Run standalone

Start the watchdog socket server first, then:

```bash
ros2 run watchdog_dummy loop_node
```

## Simulate missing heartbeats

```bash
ros2 run watchdog_dummy loop_node --ros-args -p heartbeat_enabled:=false
```

## Simulate an unhealthy node

```bash
ros2 run watchdog_dummy loop_node --ros-args \
  -p is_healthy:=false -p status_message:="test failure"
```

The watchdog config must use ID `1` and launch command:
`ros2 run watchdog_dummy loop_node`.

This package uses JSON-line messages and the FNV-1a checksum convention shown
in the accompanying watchdog example. It is not wire-compatible with a
different C++ binary serialization format.
