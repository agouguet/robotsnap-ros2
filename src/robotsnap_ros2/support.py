"""Read a RobotSNAP session off a ROS2 graph instead of off the bridge socket.

The bridge is how a session runs with nothing but this package: Unity dials a socket and the Python
side decodes the same CDR messages ``ros_tcp_endpoint`` would have carried. When the session is
already on a ROS2 graph - because the navigation method under test runs there, in a container or
not - there is no socket to own and nothing to decode. ``rclpy`` hands the messages over already
typed, and this module adapts them to the same small surface
:class:`robotsnap.client.RobotSNAPClient` presents, so the viewer draws the same window whichever
transport carried the session.

``rclpy`` is imported when a session is opened and never at module import: a machine without a ROS2
install can import this module, and only a caller that asks for a ROS2 session is told it needs one.

    source /opt/ros/humble/setup.bash
    python -m robotsnap viewer --ros2

The topics are the ones of :mod:`robotsnap.topics`, unchanged: ``/odom``, ``/scan``, ``/map``,
``/simulation/state``, ``/simulation/agents``, ``/simulation/control_result``, and
``/robot_<id>/odom`` for a fleet.

This client also *writes*, and it writes the same way the TCP bridge does: ``publish(topic,
msg_type, message)`` takes a ``rosbags`` message - the type the rest of the package already builds -
encodes it to CDR, and publishes it, so a Twist built for the socket reaches the graph unchanged.
That is what makes a run transport-agnostic: :attr:`Ros2Client.bridge` answers the whole
``RobotSNAPBridge`` surface the client façade and the environment are written against, so
``RobotSNAPClient(bridge=Ros2Client().bridge)`` is a session over the graph and nothing above it
changes.
"""

from __future__ import annotations

import importlib
import os
import time
from typing import Any

from robotsnap.bridge import codec
from robotsnap.bridge.codec import json_object
from robotsnap.bridge.state import RobotSNAPState, build_state
from robotsnap.bridge.transport import Ros2Unavailable
from robotsnap.topics import (
    CMD_VEL,
    MAP,
    ODOM,
    ODOMETRY_TYPE,
    SCAN,
    SIMULATION_AGENTS,
    SIMULATION_CONTROL,
    SIMULATION_CONTROL_RESULT,
    SIMULATION_STATE,
    TWIST_TYPE,
    base,
    robot_topic,
)

__all__ = [
    "FRESH_SECONDS",
    "Ros2Client",
    "Ros2Support",
    "Ros2Unavailable",
    "import_rclpy",
    "ros2_support",
]

#: How long a message stays "current". A session that has published nothing for this long is not
#: connected any more, which is how the viewer tells a running scene from a closed one without a
#: socket to watch.
FRESH_SECONDS = 5.0

#: How often the client looks for a fleet's own topics. Enumerating the graph costs a call per
#: topic, which is not something to do eight times a frame.
DISCOVERY_SECONDS = 0.5

#: What a per-robot odometry topic ends with: ``/robot_2/odom`` ends with the bare name behind a
#: slash. Built from the constant rather than written down, because ``topics`` is the one module that
#: names a stream.
_FLEET_ODOM_SUFFIX = "/" + base(ODOM)


def import_rclpy():
    """``rclpy`` and the message packages the session subscribes with.

    Raises :class:`Ros2Unavailable` with the line that fixes it, because the failure a reader meets
    is almost always a shell that never sourced a ROS2 install rather than a broken one.
    """
    try:
        import rclpy
        from nav_msgs.msg import OccupancyGrid, Odometry
        from sensor_msgs.msg import LaserScan
        from std_msgs.msg import String
    except ImportError as exc:  # pragma: no cover - depends on the machine
        raise Ros2Unavailable(
            "a ROS2 session needs rclpy, which comes with a ROS2 install; source one first "
            f"(source /opt/ros/humble/setup.bash) - {exc}"
        ) from exc

    return rclpy, {
        "OccupancyGrid": OccupancyGrid,
        "Odometry": Odometry,
        "LaserScan": LaserScan,
        "String": String,
    }


class Ros2Support:
    """The pieces of ``rclpy`` a transport needs, resolved once and by message name.

    A transport that stands beside the socket does two things the viewer never did: it turns a
    message name into a class the graph can publish, and it moves a message through CDR in both
    directions. Both are done here rather than in the gateway so the one place that imports
    ``rclpy`` stays the one place that knows how a ROS2 install is shaped.

    A name is resolved the way ROS2 spells it, ``pkg/msg/Type``. The short spellings the bridge
    already holds (``Odometry``) are answered from the types imported up front, and a full name is
    imported on demand - ``nav_msgs/msg/Odometry`` reaches ``nav_msgs.msg.Odometry`` - so a topic
    whose type this package never named still crosses. The two are kept apart on purpose: a full name
    that says ``other_msgs/msg/Odometry`` must never be answered with the ``nav_msgs`` class just
    because the last segment matches.
    """

    def __init__(self, rclpy, types: dict[str, Any] | None = None) -> None:
        self.rclpy = rclpy
        self._known = dict(types or {})
        self._cache: dict[str, Any] = {}

    def get_message(self, name: str):
        """The message class behind ``pkg/msg/Type``, or its short spelling.

        Raises :class:`Ros2Unavailable` when the name is not the ROS2 shape or its package holds
        no such message, because a caller that asked for a type the graph cannot carry has to hear
        it once rather than per message.
        """
        key = str(name).strip()
        if not key:
            raise Ros2Unavailable("a message type name is empty")
        cached = self._cache.get(key)
        if cached is not None:
            return cached

        if "/" not in key:
            known = self._known.get(key)
            if known is None:
                raise Ros2Unavailable(
                    f"{key!r} is a short name this package does not know; name the message as "
                    "pkg/msg/Type"
                )
            self._cache[key] = known
            return known

        parts = key.split("/")
        if len(parts) != 3 or parts[1] != "msg":
            raise Ros2Unavailable(f"{key!r} is not a message type name (pkg/msg/Type)")
        try:
            module = importlib.import_module(f"{parts[0]}.msg")
        except ImportError as exc:
            raise Ros2Unavailable(f"the graph has no message package {parts[0]!r}: {exc}") from exc
        message = getattr(module, parts[2], None)
        if message is None:
            raise Ros2Unavailable(f"the package {parts[0]!r} has no message {parts[2]!r}")
        self._cache[key] = message
        return message

    def serialize(self, message) -> bytes:
        """CDR bytes of a message, exactly as ``rosbags`` would have written them."""
        from rclpy.serialization import serialize_message

        return serialize_message(message)

    def deserialize(self, payload: bytes, message_type):
        """The message carried by ``payload``, read as ``message_type``."""
        from rclpy.serialization import deserialize_message

        return deserialize_message(bytes(payload), message_type)

    def new_executor(self):
        """An executor of this support's own, for a component that spins beside another node.

        ``rclpy.spin_once`` with no executor uses the process-wide one, and two threads spinning the
        same executor raise rather than share it. A gateway runs next to a viewer or a test node in
        the same interpreter often enough that it owns its executor instead.
        """
        from rclpy.executors import SingleThreadedExecutor

        return SingleThreadedExecutor()


def ros2_support() -> Ros2Support:
    """A :class:`Ros2Support` ready to use, importing ``rclpy`` or raising the line that fixes it."""
    rclpy, types = import_rclpy()
    return Ros2Support(rclpy, types)


class Ros2Client:
    """One ROS2 graph, read through the surface :class:`RobotSNAPClient` presents.

    The viewer reads four things from a client: the last message of a topic, the JSON body of
    ``/simulation/state`` and ``/simulation/agents``, whether a session is connected, and a
    ``bridge`` whose ``snapshot`` and ``topic_counts`` describe the world. Every one of them answers
    here from a subscription, so a reader never has to know which transport it is holding. Writing
    answers from the same place - :meth:`publish`, :meth:`send_cmd_vel` - which is what lets the
    environment and the client façade take either transport: :attr:`bridge` is a full stand-in for
    the TCP bridge, not a read-only view of one.

    It owns its node and, when nothing else had initialized ROS2, the context around it: closing the
    window leaves the process the way it found it.
    """

    def __init__(self, node_name: str = "robotsnap_viewer", *, fresh_seconds: float = FRESH_SECONDS):
        rclpy, types = import_rclpy()
        self._support = Ros2Support(rclpy, types)
        self._rclpy = rclpy
        self._types = types
        self._fresh_seconds = float(fresh_seconds)
        self._started_context = not rclpy.ok()
        if self._started_context:
            rclpy.init(args=None)

        self._error: str | None = None
        self._node = rclpy.create_node(node_name)
        self._stored: dict[str, Any] = {}
        self._counts: dict[str, int] = {}
        self._received_at: dict[str, float] = {}
        self._subscriptions: set[str] = set()
        self._publishers: dict[str, Any] = {}
        self._discovered_at = 0.0
        self._closed = False
        self._ended = False

        self._subscribe(ODOM, types["Odometry"])
        self._subscribe(SCAN, types["LaserScan"])
        self._subscribe(MAP, types["OccupancyGrid"])
        self._subscribe(SIMULATION_STATE, types["String"])
        self._subscribe(SIMULATION_AGENTS, types["String"])
        self._subscribe(SIMULATION_CONTROL_RESULT, types["String"])

        self.bridge = _BridgeView(self)

    # -- the surface the viewer reads ---------------------------------------

    @property
    def is_connected(self) -> bool:
        """True while the session has published something recently.

        A ROS2 reader has no socket to watch, so the freshness of the streams stands in for it: a
        graph that stopped speaking is a session that is not there, whether it closed or went away.
        """
        self._pump()
        now = time.monotonic()
        return any(
            now - self._received_at.get(key, float("-inf")) <= self._fresh_seconds
            for key in (base(SIMULATION_STATE), base(ODOM))
        )

    @property
    def last_error(self) -> str | None:
        return self._error

    @property
    def closed(self) -> bool:
        """True once the session is over, whether it was stopped or its context went away.

        A caller that loops - the viewer - reads this to leave: a ROS2 context shut down underneath a
        reader is the end of the session, and a window that kept polling it would spin on a graph that
        is not there any more.
        """
        return self._closed or self._ended

    @property
    def session_label(self) -> str:
        """What the viewer prints instead of "port N": a graph is addressed by its domain."""
        return f"the ROS2 graph (ROS_DOMAIN_ID={os.environ.get('ROS_DOMAIN_ID', '0')})"

    def last_message(self, topic: str):
        """The last message of a topic, or None while it has not arrived."""
        self._pump()
        return self._stored.get(base(topic))

    def snapshot(self) -> dict[str, Any] | None:
        """The ``/simulation/state`` body as a dict, which is what the viewer reads."""
        self._pump()
        body = json_object(self._stored.get(base(SIMULATION_STATE)))
        return None if body is None else dict(body)

    def agents(self) -> dict[str, Any] | None:
        """The ``/simulation/agents`` body as a dict, or None."""
        self._pump()
        body = json_object(self._stored.get(base(SIMULATION_AGENTS)))
        return None if body is None else dict(body)

    # -- writing ------------------------------------------------------------

    def publish(self, topic: str, msg_type: str, message) -> bool:
        """Publish a ``rosbags`` message on the graph, the way the TCP bridge would.

        The message is the one the rest of the package already builds - ``send_cmd_vel`` hands over a
        Twist, the client façade hands over a ``std_msgs/String`` - so it is encoded to CDR exactly as
        it would have been for the socket and read back as the class the graph knows. No field is
        copied, which is what lets one method serve both transports. Returns ``False`` and records why
        when the type is unknown or the graph refuses the publisher.
        """
        try:
            payload = codec.encode(msg_type, message)
        except codec.CodecError as exc:
            self._error = f"cannot encode {msg_type}: {exc}"
            return False
        return self.publish_payload(topic, msg_type, payload)

    def publish_payload(self, topic: str, msg_type: str, payload: bytes) -> bool:
        """Publish CDR bytes on the graph, under the message type they were written with."""
        if self._closed or self._ended:
            self._error = self._error or "the session is over"
            return False
        try:
            message_type = self._support.get_message(msg_type)
            message = self._support.deserialize(payload, message_type)
        except Exception as exc:
            self._error = f"the graph has no message type {msg_type}: {exc}"
            return False

        publisher = self._publisher_for(topic, message_type)
        if publisher is None:
            return False
        try:
            publisher.publish(message)
        except Exception as exc:
            self._error = f"could not publish on {topic}: {exc}"
            return False
        return True

    def send_cmd_vel(
        self, linear_x: float, angular_z: float, robot: str | None = None
    ) -> bool:
        """Send a ``geometry_msgs/Twist`` on the graph, on the name the robot answers to.

        ``robot`` left as ``None`` addresses ``/cmd_vel``, which reaches the primary robot; an id
        addresses ``/robot_<id>/cmd_vel``, the same rule the TCP client follows.
        """
        topic = CMD_VEL if robot is None else robot_topic(robot, CMD_VEL)
        try:
            payload = codec.encode_twist(linear_x, angular_z)
        except codec.CodecError as exc:
            self._error = f"cannot encode cmd_vel: {exc}"
            return False
        return self.publish_payload(topic, TWIST_TYPE, payload)

    def count_subscribers(self, topic: str) -> int:
        """How many nodes have subscribed to ``topic`` on the graph.

        This is what tells a client that the endpoint has heard Unity's ``__subscribe``: the endpoint
        turns each of them into a ROS2 subscription, so a count above zero is the same evidence the TCP
        bridge reads off the socket.
        """
        self._pump()
        try:
            return int(self._node.count_subscribers(topic))
        except Exception as exc:  # a graph that cannot be queried is not a reason to stop
            self._error = self._error or str(exc)
            return 0

    def _publisher_for(self, topic: str, message_type):
        """The publisher for one topic, made once - creating one waits for the graph."""
        publisher = self._publishers.get(topic)
        if publisher is not None:
            return publisher
        try:
            publisher = self._node.create_publisher(message_type, topic, 10)
        except Exception as exc:
            self._error = f"could not publish on {topic}: {exc}"
            return None
        self._publishers[topic] = publisher
        return publisher

    def wait_until_ready(self, timeout: float = 30.0) -> bool:
        """Block until the session speaks, which is what starting the viewer before Unity needs."""
        deadline = time.monotonic() + max(0.0, float(timeout))
        while time.monotonic() < deadline:
            if self.is_connected:
                return True
            self._pump(sleep=0.02)
        return self.is_connected

    def poll(self) -> None:
        """Hand the pending messages to the subscriptions without reading anything.

        Every accessor polls on its own, so the viewer never calls this. It is for a caller that has
        to keep the graph moving while it waits for something else - a test waiting for two nodes to
        pair, say - and wants that without reading a stream it is not ready for.
        """
        self._pump()

    def stop(self) -> None:
        """Destroy the node and, when this client started ROS2, shut the context down again."""
        if self._closed:
            return
        self._closed = True
        if self._ended:
            # The context is already gone: there is no node to destroy and nothing to shut down.
            return
        try:
            self._node.destroy_node()
        except Exception as exc:  # a context already shut down is not an error worth raising
            self._error = self._error or str(exc)
        finally:
            if self._started_context and self._rclpy.ok():
                self._rclpy.shutdown()

    # -- subscriptions ------------------------------------------------------

    def _subscribe(self, topic: str, message_type) -> None:
        key = base(topic)
        if key in self._subscriptions:
            return

        self._subscriptions.add(key)
        self._node.create_subscription(
            message_type, topic, lambda message, k=key: self._store(k, message), 10
        )

    def _store(self, key: str, message) -> None:
        self._stored[key] = message
        self._counts[key] = self._counts.get(key, 0) + 1
        self._received_at[key] = time.monotonic()

    def _pump(self, sleep: float = 0.0) -> None:
        """Hand the pending callbacks to the subscriptions, and follow a fleet's own topics."""
        if self.closed:
            return
        if not self._rclpy.ok():
            # A signal handler, or anything else, shut the context down: the session is over and the
            # reader says so, so the window a caller is drawing stops instead of polling a dead graph.
            self._ended = True
            return
        try:
            self._rclpy.spin_once(self._node, timeout_sec=0.0)
        except Exception as exc:
            # A context shut down underneath us - Ctrl-C, or the signal that ends a scripted run - is the
            # session ending rather than a drawing failure. The reader stops reading and the window keeps
            # its last frame, which is what a viewer owes a session that is already over.
            self._error = self._error or str(exc)
            self._ended = True
            return
        now = time.monotonic()
        if now - self._discovered_at >= DISCOVERY_SECONDS:
            self._discovered_at = now
            self._follow_the_fleet()
        if sleep:
            time.sleep(sleep)

    def _follow_the_fleet(self) -> None:
        """Subscribe to the per-robot odometry a fleet publishes under its own names.

        A scenario with several robots publishes ``/robot_2/odom`` beside the bare ``/odom``, and
        those names only exist once the roster has published them. Nothing here enumerates a fleet up
        front: a name that appears is subscribed to, and the viewer draws what it finds. A
        single-robot session never grows one, and pays a graph query twice a second for it.
        """
        try:
            topics = self._node.get_topic_names_and_types()
        except Exception as exc:  # a graph that cannot be listed is not a reason to stop drawing
            self._error = str(exc)
            return

        for name, types in topics:
            key = base(name)
            if key == base(ODOM) or not key.endswith(_FLEET_ODOM_SUFFIX):
                continue
            if key in self._subscriptions or ODOMETRY_TYPE not in types:
                continue
            self._subscribe(name, self._types["Odometry"])


class _BridgeView:
    """A :class:`Ros2Client` wearing the surface of :class:`~robotsnap.bridge.server.RobotSNAPBridge`.

    The bridge builds a typed :class:`RobotSNAPState` from the messages it decoded; this builds the
    same state from the messages the client subscribed to, through the same function, so a snapshot is
    identical on both transports. It also answers the calls the client façade and the environment make
    on a bridge - ``publish``, ``latest``, ``topic_counts``, ``wait_for_subscription`` - which is what
    makes ``RobotSNAPClient(bridge=Ros2Client().bridge)`` a session over the graph with nothing above
    it changed. ``port`` is zero: a graph is not dialled, it is joined.

    ``is_running`` starts ``False`` so the façade owns the lifetime: a client built on this starts it
    and stops it, exactly as it does for a socket.
    """

    def __init__(self, client: Ros2Client):
        self._client = client
        self._running = False

    # -- what the viewer reads ---------------------------------------------

    @property
    def port(self) -> int:
        return 0

    @property
    def session_label(self) -> str:
        """What a caller prints instead of a port; a graph is addressed by its domain."""
        return self._client.session_label

    def snapshot(self) -> RobotSNAPState:
        self._client._pump()
        return build_state(self._client._stored)

    def topic_counts(self) -> dict[str, int]:
        self._client._pump()
        return dict(self._client._counts)

    def latest(self, topic: str):
        """Last message of ``topic``, or None. The rosbags form is not available over a graph."""
        return self._client.last_message(topic)

    def latest_raw(self, topic: str) -> bytes | None:
        """CDR bytes of the last message of ``topic``, or None before the first one."""
        message = self._client.last_message(topic)
        if message is None:
            return None
        try:
            return self._client._support.serialize(message)
        except Exception as exc:
            self._client._error = self._client._error or str(exc)
            return None

    @property
    def last_error(self) -> str | None:
        return self._client.last_error

    @property
    def is_connected(self) -> bool:
        return self._client.is_connected

    # -- what the client façade calls --------------------------------------

    def publish(self, topic: str, msg_type: str, message) -> bool:
        """Send one message, taking the same ``rosbags`` instance the TCP bridge takes."""
        return self._client.publish(topic, msg_type, message)

    def wait_for_connection(self, timeout: float | None = None) -> bool:
        return self._client.wait_until_ready(30.0 if timeout is None else timeout)

    def wait_for_subscription(self, topic: str, timeout: float | None = None) -> bool:
        """Block until something on the graph listens to ``topic``.

        The endpoint turns Unity's ``__subscribe`` into a ROS2 subscription, so a subscriber count
        above zero is the graph's version of the evidence the TCP bridge reads off the socket. Waiting
        for it is what stops a command from being dropped by a session that has not registered yet.
        """
        deadline = None if timeout is None else time.monotonic() + max(0.0, float(timeout))
        while True:
            if self._client.count_subscribers(topic) > 0:
                return True
            if deadline is not None and time.monotonic() >= deadline:
                return False
            self._client._pump(sleep=0.02)

    # -- lifetime -----------------------------------------------------------

    @property
    def is_running(self) -> bool:
        return self._running

    def start(self) -> None:
        self._running = True

    def stop(self) -> None:
        self._running = False
        self._client.stop()
