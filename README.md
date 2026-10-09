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

## Transforms

Unity names the frames its streams sit in but publishes no `/tf`, and a graph reader with only frame
names and no links draws nothing: RViz2 cannot place the map, the robot or the lidar, and a
navigation stack sees three unconnected streams. The mirror derives the tree from the messages
themselves and publishes it:

```
map --static--> odom --dynamic--> base_link --static--> laser
```

- `odom -> base_link` is the robot's pose, on `/tf` at the odometry rate, read straight from
  `nav_msgs/Odometry` (`pose.pose`, `child_frame_id`) and stamped with its header.
- `map -> odom` and `base_link -> laser` are identity links, on `/tf_static` once both frames they
  join have been seen. `/tf_static` is latched (`transient_local`, `reliable`, depth 1), so RViz2
  opened late, or a navigation stack started after the map, still receives them.

Every frame is normalised before publication: a leading `/` is removed and the name is trimmed,
because tf2 rejects a frame id that starts with a slash - and the session does spell its frames that
way (`/map`, `/odom`). A robot in a fleet gets its own tree off its own namespace: `/robot_2/odom`
and `/robot_2/scan` give `map -> robot_2/odom` and `robot_2/base_link -> robot_2/laser`.

Two things worth knowing:

- **The application has priority.** When the session publishes its own `/tf` or `/tf_static`, the
  mirror republishes it like any other stream and does not synthesise that topic - a Unity build that
  already broadcasts a tree, or a scenario with a fixed calibration, keeps it untouched.
- **The lidar link is flat.** `base_link -> laser` carries zero translation, so the sensor sits at
  the robot's origin. In a top-down 2D view (RViz2, Nav2's costmaps) that is the same picture, since
  only x, y and yaw are read; a sensor mounted high or offset in 3D would need a real calibration.

## How the core finds it

The core knows nothing about ROS2. It owns one entry-point group, `robotsnap.transports`, and
reads it through `robotsnap.bridge.transport.load_ros2()`. This package registers itself there:

    [project.entry-points."robotsnap.transports"]
    ros2 = "robotsnap_ros2"

When the plugin is installed, `--ros2`, `--transport ros2` and the ROS2 viewer find this module;
when it is not, the core refuses with the one line that installs it, and every other mode keeps
working.
