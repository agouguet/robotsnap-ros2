# Unity, the Python viewer, and a navigation method in a container

Three processes, three questions: who serves Unity, where the navigation method
runs, and how the Python viewer reads the session. This page answers them in the
order they depend on each other.

## The one constraint that decides everything

**Unity dials exactly one endpoint, on one port.** The simulator is an
`ROS-TCP-Connector` client: it opens a TCP connection to port 10000 (by default)
and speaks the connector's protocol. Whoever holds that port is the whole graph
as far as Unity is concerned, and there are only two things that can hold it:

| Server on 10000 | What Unity is attached to | Python side |
| --- | --- | --- |
| `robotsnap bridge` (this package) | a socket owned by Python | `robotsnap.client`, the Gymnasium environment, the viewer |
| `ros2 run ros_tcp_endpoint default_server_endpoint` | a ROS2 graph (DDS) | `rclpy`, `rviz2`, and `robotsnap watch --ros2` |

Two listeners on one port is one too many, so these two never run together.
Which one you pick is the transport question, and this package answers both
sides of it: it can serve Unity itself, and - with `--ros2` - put the very same
session on a ROS2 graph while it does, so a ROS2 navigation method sees Unity's
streams without `ros_tcp_endpoint` being involved at all. The rest of the Python
package never learns which transport is in play: the environment, the client and
the viewer read the bridge's own surface whichever one was chosen.

## Layout 0 - this package is the endpoint, and the graph is downstream

This is the layout to prefer when the navigation method is a ROS2 node, whether
it runs in a container or not, and the RL side is this package. One process holds
Unity's socket and mirrors the session both ways.

```
      /scan /odom /simulation/state        Unity
  <------------------------------------>  (dials :10000)
   robotsnap bridge --ros2   (host, port 10000)
      |  publishes every stream Unity announces
      v
   ROS2 graph  <-->  docker: nav method (ROS2 node)
   ^        |
   |        +--> /cmd_vel, /simulation/control written back to Unity
   |
   +-- the same process also serves robotsnap.client, the Gymnasium env, the viewer
```

```bash
# one process: Unity's endpoint and the ROS2 mirror, in-process
source /opt/ros/humble/setup.bash
python -m robotsnap bridge --ros2          # or: robotsnap bridge --ros2

# the navigation method, in its container, on the same graph
docker run --rm -it --network host \
    -e ROS_DOMAIN_ID=0 -e RMW_IMPLEMENTATION=rmw_fastrtps_cpp \
    your-nav-image
```

Which topics cross is not written down anywhere: it is what Unity announces.
A stream the simulator publishes is published on the graph under the same name,
and a stream it subscribes to is subscribed on the graph and written back on the
socket under the name Unity registered - `/cmd_vel` on the primary robot, and
`/robot_<id>/cmd_vel` for a fleet, the same rule
`robotsnap.bridge.server.destination_for` applies. A build that adds a topic gets
it bridged without this package changing.

The mirror is a translation of placement, not of content: Unity's payload is CDR
and is deserialized once to be published, so no field is copied and nothing can
drift on the way through. `robotsnap bridge --ros2` prints one line per skipped
topic when the graph has no message type for it, and keeps the session running.

### The run itself can ride the graph too

The PyTorch side is a client like any other, and it can be handed either
transport. `--transport ros2` joins the graph instead of binding the port, so the
same environment, the same policies and the same metrics drive a session served
by `ros_tcp_endpoint` - or by `robotsnap bridge --ros2` above, or by any other
endpoint that speaks the same topics:

```bash
# Unity is served by the ROS2 endpoint, and the training run is a node on the graph
source /opt/ros/humble/setup.bash
ros2 run ros_tcp_endpoint default_server_endpoint --ros-args -p ROS_IP:=0.0.0.0 -p ROS_TCP_PORT:=10000
python -m robotsnap train --transport ros2 --algo ppo --timesteps 20000 --keep

# inference the same way, at x1, next to the method under test
python -m robotsnap play --transport ros2 --load policy.zip --time-scale 1
```

| | `--transport tcp` (default) | `--transport ros2` |
| --- | --- | --- |
| Who holds port 10000 | this package | `ros_tcp_endpoint`, `robotsnap bridge --ros2`, ... |
| How the run reaches the session | the bridge's own socket | the graph |
| `--host` / `--port` | used | meaningless; the graph is addressed by `ROS_DOMAIN_ID` |
| Bootstrapping a scenario | `robotsnap.client` writes it and asks Unity to load it | identical, over `/simulation/control` |
| Commands, metrics, pacing | unchanged | unchanged |

Nothing above the client changes: `robotsnap.client.open_client(transport=...)`
returns the same façade, and `RobotSNAPClient.send_cmd_vel`, `play`,
`launch_scenario`, `metrics_summary` and the rest keep their contract. The
environment is handed one or the other and does not know which.

Readiness is the one thing that is spelled differently. Over a socket the client
waits for Unity's `__subscribe` frame; over the graph it waits for a subscriber to
appear on `/simulation/control`, which is what the endpoint turns that frame
into. Both mean the same thing - a command written before it would be answered
by nobody.

### Watching any of these

The 2D window is an option of every command, so a run never needs a second
terminal: `--viewer` (whose older spelling `--render` still works) draws the
session while the run goes on.

```bash
python -m robotsnap train --transport ros2 --viewer --algo ppo --timesteps 20000
python -m robotsnap bench --viewer --steps 200
python -m robotsnap bridge --ros2 --viewer      # the bridge draws what it serves
python -m robotsnap watch --ros2                # and watch alone drives nothing
```

`watch` (still accepted as `viewer`) is the only command that drives nothing: it
is what you run to look at a session Unity or an external ROS2 node is driving.

## Layout A - `ros_tcp_endpoint` serves Unity, the method joins the graph

The same picture as Layout 0, with the endpoint kept as a separate process. Take
this one when the ROS2 endpoint is already part of your setup, when you want
Unity served by a process that knows nothing about this package, or when the
interpreter that runs the viewer has no bridge of its own. Unity is served by the
ROS2 endpoint, the method joins the same graph, and the viewer reads the graph.

```
                /scan /odom /simulation/state      Unity
   ROS2 graph  <------------------------------>  (dials :10000)
        ^  |          ros_tcp_endpoint (host, port 10000)
        |  v
   +----+----------------+     +-----------------------+
   | docker: nav method  |     | robotsnap watch       |
   |   (ROS2 node)       |     |   --ros2              |
   |  /cmd_vel out       |     |  /simulation/* in     |
   +---------------------+     +-----------------------+
```

```bash
# 1. the ROS2 side: Unity's server, on the host
source /opt/ros/humble/setup.bash
ros2 run ros_tcp_endpoint default_server_endpoint --ros-args -p ROS_IP:=0.0.0.0 -p ROS_TCP_PORT:=10000

# 2. the navigation method, in its container, on the same graph
docker run --rm -it --network host \
    -e ROS_DOMAIN_ID=0 -e RMW_IMPLEMENTATION=rmw_fastrtps_cpp \
    your-nav-image

# 3. the viewer, reading the graph rather than owning the socket
source /opt/ros/humble/setup.bash
python -m robotsnap watch --ros2

# and, if you want ROS2's own tools on the same graph
ros2 topic echo /scan --once
rviz2
```

Then press Play in Unity. Nothing has to be started in a particular order beyond
the endpoint being up before Unity dials it.

### Container networking, the part that actually bites

The graph is discovered by multicast, so the container has to be on a network
where it can see the host's ROS2 traffic. Two ways, in order of least surprise:

- **`--network host`** (what the command above does). The container shares the
  host's network namespace, so DDS discovery behaves as if the node ran on the
  host. Linux only, and it is the least fiddly.
- **A docker bridge network, with the endpoint inside it too.** This is what a
  compose file with `networks: [ros]` does, and it works because the endpoint
  container and the method container are on the same L2 segment: they discover
  each other, and the host reaches the endpoint through its published port. It
  stops working the moment one of the two is *outside* the compose network.

Two environment variables have to match or the two sides are invisible to each
other, with no error to say so:

- `ROS_DOMAIN_ID` (default `0`). A different id is a different graph.
- `RMW_IMPLEMENTATION` (`rmw_fastrtps_cpp`, `rmw_cyclonedds_cpp`, ...). Two
  different middlewares on one host do not talk without a discovery server.

If the method cannot join the graph, check those two before anything else:

```bash
ros2 topic list                       # on the host: is /scan there?
docker exec -it <container> bash -lc 'source /opt/ros/humble/setup.bash && ros2 topic list'
```

## Layout B - the method is ROS1

A ROS1 method (CADRL, SARL\*, HATEB, DRL\_VO, ...) needs the ROS1 bridge, and the
shape is the same plus one node:

```
[docker: roscore] <-> [docker: ros1_bridge dynamic_bridge] <-> ROS2 graph
                                                                 ^
                                          ros_tcp_endpoint (host) + watch --ros2
```

The `ros1_bridge docker dynamic_bridge --bridge-all-topics` publishes every ROS1
topic into ROS2, so the ROS2 side - and therefore the viewer and Unity - sees the
same streams. Only the ROS2 half is on the host; the ROS1 half must not be,
because ROS1 has no DDS to join.

## Layout C - no ROS2 at all

When the navigation method is Python (this package's environment, a policy
checkpoint, a script), there is no graph to join and no reason to have one:

```bash
python -m robotsnap watch --port 10000 --rate 30    # owns the bridge Unity dials
python examples/env_random_episode.py --port 10000  # same bridge, same session
```

The method then steers the robot through `robotsnap.client` - `send_cmd_vel`,
`set_robot_goal`, `launch_scenario` - instead of publishing `/cmd_vel` itself.

## What the viewer needs from the graph

`--ros2` subscribes to the same five streams the bridge decodes, and to nothing
else. A reader that has all of them draws the whole window:

| Topic | Type | What the viewer draws from it |
| --- | --- | --- |
| `/odom` | `nav_msgs/Odometry` | the robot, its heading and the scan's placement |
| `/scan` | `sensor_msgs/LaserScan` | the lidar points |
| `/map` | `nav_msgs/OccupancyGrid` | the occupancy grid |
| `/simulation/state` | `std_msgs/String` (JSON) | run state, clock, scenario, crowd, goal |
| `/simulation/agents` | `std_msgs/String` (JSON) | the lidar-visibility flag of each crowd member |

`/robot_<id>/odom` is picked up as soon as it appears, so a fleet draws its other
robots from their own odometry; a session with one robot never grows one.

Nothing in the package publishes: the navigation method owns the commands, and
the viewer only reads. To let the method steer, put the robot under ROS control
(`/simulation/control` with `set_control_mode ros`, or the control-mode selector
in the HUD), or the scenario keeps driving it.

## Known friction

- **Running the test suite with ROS2 sourced fails before it starts.** ROS2
  installs two pytest entry points (`launch_testing`, and the ROS one) that a
  plain virtualenv cannot import. Either run the tests without sourcing ROS2, or
  disable plugin autoloading:

  ```bash
  PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest
  ```

  The ROS2 transport tests skip themselves when `rclpy` is not importable, so the
  suite is honest on both kinds of machine.
- **`--ros2` and `--host`/`--port` are not both meaningful.** A graph is
  addressed by its domain id, not by a host and a port; the viewer says so in its
  status line rather than printing a port that means nothing.
- **A paused session still reads as connected** for a few seconds, because a ROS2
  reader has no socket to watch and uses the freshness of the streams instead.

## Comparer une politique Python à une méthode ROS2

`benchmark --policy external` (alias `ros2`) mesure une méthode qui conduit le
robot *depuis l'extérieur de Python* : un nœud ROS2, éventuellement dans un
docker, publie `cmd_vel` dans le graphe du bridge pendant que Python fait tourner
la suite de scénarios, cadencer le monde en lockstep et récolte les métriques. En
mode externe l'environnement n'écrit aucune commande - sinon la sienne entrerait
en course avec celle du nœud - mais il relâche le pacing, attend la période de
contrôle et lit le monde exactement comme pour une politique Python : la même
suite de scénarios et le même tableau de métriques servent donc aux deux, ce qui
est tout l'intérêt de la comparaison.

Séquence :

1. Lancer Unity sur la scène, comme pour un run Python habituel.
2. Démarrer le bridge en transport ROS2 :

   ```bash
   python -m robotsnap.bridge --ros2
   ```

3. Lancer le nœud docker de la méthode : il s'abonne à la même grappe (`/odom`,
   `/scan`, `/map`, `/simulation/state`, `/simulation/agents`) et publie sa
   vitesse sur `cmd_vel`, **dans le même repère que le bridge** (vitesse linéaire
   et angulaire du robot, cap de la scène, pas un repère de nœud propre).
4. Mesurer la méthode externe, puis la politique Python, sur la même suite :

   ```bash
   python -m robotsnap benchmark --policy external --suite basic --episodes 5 --out results/ros2.json
   python -m robotsnap benchmark --policy scripted --suite basic --episodes 5 --out results/scripted.json
   ```

5. Comparer les deux campagnes sauvegardées :

   ```bash
   python -m robotsnap benchmark --compare results/scripted.json results/ros2.json
   ```

Une politique RL se compare de la même façon : `--load <checkpoint>` au lieu de
`--policy scripted`, la suite et les métriques restant identiques.
