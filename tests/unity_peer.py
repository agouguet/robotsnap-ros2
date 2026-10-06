"""One fake Unity peer, shared by the bridge and the client test suites.

Both suites need the same thing - a TCP client that speaks the ROS-TCP-Connector
framing in ROS2 mode - so there is one implementation of it, and both import it
from here (``tests`` is on ``pythonpath``, see ``pyproject.toml``).

The framing is what ``ROSConnection`` writes with ``ROS2`` defined:

- a string is ``[int32 len(utf8) + 1][utf8 bytes][0x00]``, the declared length
  including the trailing NUL;
- a system command is ``[string command][int32 len(json) + 1][json][0x00]``,
  the command name as the frame destination and the body as the payload, with
  the name *not* repeated inside the body;
- a topic message is ``[string topic][int32 len(payload)][CDR payload]``, and
  the payload carries no extra NUL.
"""

import json
import math
import socket
import struct
import threading

import numpy as np

from robotsnap import topics
from robotsnap.bridge import codec, protocol

#: How long a socket operation may take before a test is considered hung.
CONNECTION_TIMEOUT = 2.0

_STRING = "std_msgs/msg/String"
_CONTROL = "simulation/control"

#: The four commands of the contract that take an optional ``"robot"`` key.
_ROBOT_COMMANDS = (
    "set_robot_goal",
    "clear_robot_goal",
    "stop_robot",
    "set_control_mode",
)


def unity_string(text):
    """ROS2 string encoding used by the Unity connector, trailing NUL included."""
    raw = text.encode("utf-8")
    return struct.pack("<I", len(raw) + 1) + raw + b"\x00"


def unity_command_frame(command, params):
    """A system command frame exactly as Unity writes it in ROS2 mode."""
    body = json.dumps(params).encode("utf-8")
    return unity_string(command) + struct.pack("<I", len(body) + 1) + body + b"\x00"


def unity_topic_frame(topic, payload):
    """A topic frame exactly as Unity writes it in ROS2 mode."""
    payload = bytes(payload)
    return unity_string(topic) + struct.pack("<I", len(payload)) + payload


def _time_message(seconds: float):
    """A ``builtin_interfaces/Time`` for ``seconds`` since the epoch."""
    whole = int(seconds)
    nanos = int(round((seconds - whole) * 1e9))
    if nanos >= 1_000_000_000:
        whole, nanos = whole + 1, nanos - 1_000_000_000
    return codec.TYPESTORE.types["builtin_interfaces/msg/Time"](sec=whole, nanosec=nanos)


def _header(frame_id: str, stamp: float):
    return codec.TYPESTORE.types["std_msgs/msg/Header"](
        stamp=_time_message(stamp), frame_id=frame_id
    )


def _point(x: float, y: float, z: float = 0.0):
    return codec.TYPESTORE.types["geometry_msgs/msg/Point"](x=float(x), y=float(y), z=float(z))


def _yaw_quaternion(yaw: float):
    return codec.TYPESTORE.types["geometry_msgs/msg/Quaternion"](
        x=0.0, y=0.0, z=math.sin(yaw / 2.0), w=math.cos(yaw / 2.0)
    )


def _pose(x: float, y: float, yaw: float = 0.0):
    return codec.TYPESTORE.types["geometry_msgs/msg/Pose"](
        position=_point(x, y), orientation=_yaw_quaternion(yaw)
    )


def unity_odometry(
    x: float = 0.0,
    y: float = 0.0,
    yaw: float = 0.0,
    linear_x: float = 0.0,
    angular_z: float = 0.0,
    frame_id: str = "/map",
    stamp: float = 0.0,
):
    """A ``nav_msgs/Odometry`` in the ROS frame, as the publisher writes it."""
    types = codec.TYPESTORE.types
    covariance = np.zeros(36, dtype=np.float64)
    return types["nav_msgs/msg/Odometry"](
        header=_header(frame_id, stamp),
        child_frame_id="base_link",
        pose=types["geometry_msgs/msg/PoseWithCovariance"](
            pose=_pose(x, y, yaw),
            covariance=covariance,
        ),
        twist=types["geometry_msgs/msg/TwistWithCovariance"](
            twist=types["geometry_msgs/msg/Twist"](
                linear=types["geometry_msgs/msg/Vector3"](
                    x=float(linear_x), y=0.0, z=0.0
                ),
                angular=types["geometry_msgs/msg/Vector3"](
                    x=0.0, y=0.0, z=float(angular_z)
                ),
            ),
            covariance=covariance,
        ),
    )


def unity_laserscan(
    ranges,
    angle_min: float = -math.pi,
    angle_increment: float = 0.01,
    range_min: float = 0.05,
    range_max: float = 10.0,
    frame_id: str = "/robot_1/lidar",
    stamp: float = 0.0,
):
    """A ``sensor_msgs/LaserScan`` whose beam ``i`` sits at ``angle_min + i * inc``."""
    values = [float(value) for value in ranges]
    return codec.TYPESTORE.types["sensor_msgs/msg/LaserScan"](
        header=_header(frame_id, stamp),
        angle_min=float(angle_min),
        angle_max=float(angle_min) + len(values) * float(angle_increment),
        angle_increment=float(angle_increment),
        time_increment=0.0,
        scan_time=0.0,
        range_min=float(range_min),
        range_max=float(range_max),
        ranges=np.array(values, dtype=np.float32),
        intensities=np.zeros(len(values), dtype=np.float32),
    )


def unity_occupancy_grid(
    cells,
    resolution: float = 0.05,
    origin_x: float = 0.0,
    origin_y: float = 0.0,
    frame_id: str = "/map",
    stamp: float = 0.0,
):
    """A ``nav_msgs/OccupancyGrid`` from a ``(height, width)`` grid in ROS order."""
    types = codec.TYPESTORE.types
    rows = [list(row) for row in cells]
    height = len(rows)
    width = len(rows[0]) if rows else 0
    flat = [int(value) for row in rows for value in row]
    return types["nav_msgs/msg/OccupancyGrid"](
        header=_header(frame_id, stamp),
        info=types["nav_msgs/msg/MapMetaData"](
            map_load_time=_time_message(stamp),
            resolution=float(resolution),
            width=width,
            height=height,
            origin=_pose(origin_x, origin_y),
        ),
        data=np.array(flat, dtype=np.int8),
    )


class UnityPeer:
    """Unity stand-in: registers topics, publishes state and answers commands.

    ``commands`` records ``(topic, parsed body)`` for every control frame the
    peer read, so a test can assert on the exact wire body the client produced.
    The peer answers like the simulator does: every command goes back on
    ``/simulation/control_result`` under the name it was sent with, and a
    ``humans`` command also carries the ids the scene does not hold, which the
    peer learns from the state snapshots it published.
    """

    def __init__(self, port):
        self.sock = socket.create_connection(
            ("127.0.0.1", port), timeout=CONNECTION_TIMEOUT
        )
        self.commands: list[tuple[str, dict]] = []
        #: Every frame this peer read that was not a system command, so a test
        #: can look at what a client wrote on a topic - ``/cmd_vel`` included.
        self.inbound: list[tuple[str, bytes]] = []
        self.known_human_ids: set[int] = set()
        #: Robots the published snapshots held, and the one legacy names reach.
        self.known_robot_ids: set[str] = {"robot_1"}
        self.primary_robot_id = "robot_1"
        #: Episodes the peer's fake metrics store holds, so a test can drive the same
        #: list/get/clear surface the real Unity store answers.
        self.episodes: list[dict] = []
        self._lock = threading.Lock()
        self._closed = False
        self._responder: threading.Thread | None = None

    # -- outgoing frames (peer -> bridge) ----------------------------------

    def send_command(self, command, params):
        with self._lock:
            self.sock.sendall(unity_command_frame(command, params))

    def handshake(self, version="1.0", metadata=None):
        body = json.dumps(metadata if metadata is not None else {})
        self.send_command("__handshake", {"version": version, "metadata": body})

    def register_publisher(self, topic, message_name):
        self.send_command("__publish", {"topic": topic, "message_name": message_name})

    def register_subscriber(self, topic, message_name):
        self.send_command("__subscribe", {"topic": topic, "message_name": message_name})

    def publish(self, topic, payload):
        with self._lock:
            self.sock.sendall(unity_topic_frame(topic, payload))

    def publish_string(self, topic, text):
        """Publish ``text`` as a ``std_msgs/String`` body."""
        message = codec.TYPESTORE.types[_STRING](data=text)
        self.publish(topic, codec.encode(_STRING, message))

    def publish_state(self, state):
        """Publish a ``/simulation/state`` JSON snapshot.

        The session snapshot always carries the ``robots`` roster and its
        ``robot_count`` next to the legacy keys, as the contract says. A state
        that already lists its robots keeps them; one that does not, like the
        older single-robot body, gets a one-entry roster built from its
        top-level ``robot`` key.
        """
        payload = self._state_payload(state)
        self.publish_string("/simulation/state", json.dumps(payload))
        self._learn_robots(payload.get("robots"))
        humans = payload.get("humans") if isinstance(payload, dict) else None
        if isinstance(humans, list):
            self.known_human_ids = {
                int(human["id"])
                for human in humans
                if isinstance(human, dict) and human.get("id") is not None
            }

    def _state_payload(self, state) -> dict:
        """Return ``state`` with the roster keys the contract requires."""
        payload = dict(state) if isinstance(state, dict) else {}
        robots = payload.get("robots")
        if not isinstance(robots, list):
            robots = self._legacy_roster(payload)
        payload["robots"] = robots
        payload["robot_count"] = len(robots)
        return payload

    def _legacy_roster(self, state: dict) -> list[dict]:
        """One roster entry built from the old top-level ``robot`` key."""
        robot = state.get("robot")
        if not isinstance(robot, dict):
            return []
        return [
            {
                "id": self.primary_robot_id,
                "type": "",
                "is_primary": True,
                "x": robot.get("x", 0.0),
                "y": robot.get("y", 0.0),
                "z": robot.get("z", 0.0),
                "yaw": robot.get("yaw", 0.0),
                "has_goal": bool(state.get("robot_has_goal")),
                "goal": state.get("robot_goal"),
                "start_pose": None,
                "target_pose": None,
            }
        ]

    def _learn_robots(self, robots) -> None:
        """Remember the roster a snapshot published, to answer robot commands."""
        if not isinstance(robots, list):
            return
        ids = [
            str(entry["id"])
            for entry in robots
            if isinstance(entry, dict) and entry.get("id") is not None
        ]
        self.known_robot_ids = set(ids)
        for entry in robots:
            if (
                isinstance(entry, dict)
                and entry.get("is_primary")
                and entry.get("id") is not None
            ):
                self.primary_robot_id = str(entry["id"])
                break
        else:
            if ids:
                self.primary_robot_id = ids[0]

    def publish_agents(self, body):
        """Publish a ``/simulation/agents`` JSON body."""
        self.publish_string("/simulation/agents", json.dumps(body))

    def publish_odom(self, x=0.0, y=0.0, yaw=0.0, linear_x=0.0, angular_z=0.0, stamp=0.0):
        """Publish one ``nav_msgs/Odometry``, in the ROS frame."""
        self.publish(
            "/odom",
            codec.encode(
                topics.ODOMETRY_TYPE,
                unity_odometry(
                    x=x, y=y, yaw=yaw, linear_x=linear_x, angular_z=angular_z, stamp=stamp
                ),
            ),
        )

    def publish_scan(
        self,
        ranges,
        angle_min=-math.pi,
        angle_increment=0.01,
        range_min=0.05,
        range_max=10.0,
        stamp=0.0,
    ):
        """Publish one ``sensor_msgs/LaserScan``."""
        self.publish(
            "/scan",
            codec.encode(
                topics.LASER_SCAN_TYPE,
                unity_laserscan(
                    ranges,
                    angle_min=angle_min,
                    angle_increment=angle_increment,
                    range_min=range_min,
                    range_max=range_max,
                    stamp=stamp,
                ),
            ),
        )

    def publish_map(self, cells, resolution=0.05, origin_x=0.0, origin_y=0.0, stamp=0.0):
        """Publish one ``nav_msgs/OccupancyGrid``, from a ``(height, width)`` grid."""
        self.publish(
            "/map",
            codec.encode(
                topics.OCCUPANCY_GRID_TYPE,
                unity_occupancy_grid(
                    cells,
                    resolution=resolution,
                    origin_x=origin_x,
                    origin_y=origin_y,
                    stamp=stamp,
                ),
            ),
        )

    def publish_reset_done(self):
        """Publish the ``/reset_done`` handshake Unity sends once a world is ready."""
        message = codec.TYPESTORE.types[topics.BOOL_TYPE](data=True)
        self.publish("/reset_done", codec.encode(topics.BOOL_TYPE, message))

    def publish_episode(self, episode):
        """Publish one finished episode on ``/simulation/metrics``, as Unity does."""
        self.publish_string("/simulation/metrics", json.dumps(episode))

    # -- incoming frames (bridge -> peer) ----------------------------------

    def recv_frame(self):
        return protocol.read_frame(self.sock)

    def recv_handshake(self):
        """Read the endpoint's opening frame and parse it as the handshake."""
        destination, payload = protocol.read_frame(self.sock)
        assert destination == "__handshake"
        return protocol.parse_handshake(payload)

    # -- command answering -------------------------------------------------

    def start_responder(self, sim_time_seconds=12.5):
        """Answer every control command in a background thread, as Unity does.

        The thread reads frames until the socket closes or the test ends, records
        each control body and replies on ``/simulation/control_result``.
        """
        self._responder = threading.Thread(
            target=self._respond_loop,
            args=(sim_time_seconds,),
            name="fake-unity-responder",
            daemon=True,
        )
        self._responder.start()
        return self._responder

    def _respond_loop(self, sim_time_seconds):
        self.sock.settimeout(0.2)
        while not self._closed:
            try:
                destination, payload = protocol.read_frame(self.sock)
            except socket.timeout:
                continue
            except (protocol.ProtocolError, OSError):
                return
            if destination.startswith("__"):
                continue
            key = destination.strip("/")
            with self._lock:
                # Everything a client wrote on a topic, command or not, so a test
                # can read back the /cmd_vel a control loop produced.
                self.inbound.append((destination, payload))
            if key != _CONTROL:
                continue
            body = self._parse_body(payload)
            with self._lock:
                # The destination is kept verbatim, leading slash included, so a
                # test sees exactly what the client put on the wire.
                self.commands.append((destination, body))
            command = body.get("command", "")
            unknown_ids = self._unknown_ids(body) if command == "humans" else None
            unknown_robot = (
                self._unknown_robot(body) if command in _ROBOT_COMMANDS else None
            )
            ok = not unknown_ids and unknown_robot is None
            if unknown_ids:
                message = f"unknown ids: {unknown_ids}"
            elif unknown_robot is not None:
                message = (
                    f"unknown robot {unknown_robot}; known ids: "
                    + ", ".join(sorted(self.known_robot_ids))
                )
            else:
                message = ""
            result = {
                "command": command,
                "ok": ok,
                "message": message,
                "sim_time_seconds": sim_time_seconds,
            }
            if unknown_ids is not None:
                result["unknown_ids"] = unknown_ids
            if command in _ROBOT_COMMANDS:
                # The contract echoes the id the simulator acted on.
                result["robot"] = self._acted_on_robot(body)
            payload = self._metrics_payload(command, body)
            if payload is not None:
                result["payload"] = payload
            self.publish_string(
                "/simulation/control_result",
                json.dumps(result),
            )
            if command in ("reset", "load_scenario"):
                # Unity publishes its /reset_done handshake once the applied world
                # is ready, which is the signal a reset waits for.
                self.publish_reset_done()

    def _acted_on_robot(self, body) -> str:
        """The id a robot command would act on: the named one, else the primary."""
        requested = body.get("robot")
        if isinstance(requested, str) and requested.strip():
            return requested.strip()
        return self.primary_robot_id

    def _metrics_payload(self, command: str, body: dict):
        """The document a metrics command is answered with, or ``None`` for every other command.

        The peer keeps the same three questions Unity's router answers - list the session, get one episode,
        clear the list - so the client's metrics surface is exercised through the real wire shape rather than
        against a stub of it.
        """
        if command == "metrics_episodes":
            return {
                "session": "s_fake",
                "started_at": "2026-09-30T10:00:00.0000000Z",
                "count": len(self.episodes),
                "episodes": list(self.episodes),
            }
        if command == "metrics_episode":
            identifier = str(body.get("id", ""))
            episode = next(
                (entry for entry in self.episodes if str(entry.get("id")) == identifier),
                None,
            )
            return {"id": identifier, "found": episode is not None, "episode": episode}
        if command == "metrics_clear":
            cleared = len(self.episodes)
            self.episodes = []
            return {"cleared": cleared, "session": "s_fake_new"}
        return None

    def _unknown_robot(self, body) -> str | None:
        """The named robot id the published roster does not carry, else ``None``."""
        requested = body.get("robot")
        if requested is None:
            return None
        name = str(requested).strip()
        if not name or name in self.known_robot_ids:
            return None
        return name

    def _unknown_ids(self, body) -> list[int]:
        """Ids of a ``humans`` command that the published snapshots never held."""
        commands = body.get("commands")
        if not isinstance(commands, list):
            return []
        return sorted(
            {
                int(entry["id"])
                for entry in commands
                if isinstance(entry, dict)
                and entry.get("id") is not None
                and int(entry["id"]) not in self.known_human_ids
            }
        )

    @staticmethod
    def _parse_body(payload):
        try:
            return json.loads(codec.decode(_STRING, payload).data)
        except (codec.CodecError, ValueError, AttributeError) as exc:
            return {"undecodable": str(exc)}

    def close(self):
        self._closed = True
        try:
            self.sock.close()
        except OSError:
            pass
