# RobotSNAP-ROS2

The ROS2 transport of RobotSNAP: the part that lives beside the core package rather than inside it.

Two things ship here. `robotsnap_ros2.Ros2Client` reads a session off a running ROS2 graph
instead of off the bridge socket, so the viewer, the client facade and the Gymnasium environment
draw and drive a session that no socket owns. `robotsnap_ros2.Ros2Gateway` is the other side:
`python -m robotsnap.bridge --ros2` runs it beside the TCP endpoint, mirroring the session onto a
graph so a navigation method that runs there reads the same streams and writes `/cmd_vel`.

## Installing

Install it next to the core, in the same interpreter:

    pip install -e .

`rclpy` is not a dependency: it comes with a ROS2 install, which you source before a run
(`source /opt/ros/humble/setup.bash`). This package imports it lazily, so a machine without ROS2
installs and imports the core unharmed.

## Bench a method, without the learning stack

Benchmarking a navigation method that runs on the graph needs no PyTorch and no display - the
core's extra does the rest:

    pip install "RobotSNAP[env]" robotsnap-ros2

    # Unity runs, `python -m robotsnap.bridge --ros2` mirrors the session, the
    # navigation node drives the robot, and RobotSNAP scores the suite:
    python -m robotsnap benchmark --policy external --suite basic --out results/ros2.json

`[env]` brings the Gymnasium interface the episode loop is built on; `[viewer]` is only needed if
you also want the window.

## How the core finds it

The core knows nothing about ROS2. It owns one entry-point group, `robotsnap.transports`, and
reads it through `robotsnap.bridge.transport.load_ros2()`. This package registers itself there:

    [project.entry-points."robotsnap.transports"]
    ros2 = "robotsnap_ros2"

When the plugin is installed, `--ros2`, `--transport ros2` and the ROS2 viewer find this module;
when it is not, the core refuses with the one line that installs it, and every other mode keeps
working.
