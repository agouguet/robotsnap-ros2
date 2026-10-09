"""The transform tree on a real ROS2 graph, with a fake Unity on the socket.

The unit suite composes the tree from decoded messages and proves the wiring against a stand-in
``rclpy``; this one proves the last step, that the message built here is the one a real subscriber
receives - the CDR of a ``tf2_msgs/TFMessage``, the latched QoS of ``/tf_static``, and the frame
names a reader looks them up by. It is skipped when no ROS2 install is sourced, for the same reason
``test_ros2_gateway_live.py`` is.

    source /opt/ros/humble/setup.bash
    PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/test_ros2_tf_live.py -q

The graph is shared with whatever else runs on this machine, so nothing here asserts on "the next
message": arrivals are filtered by frame, and every wait is bounded.
"""

from __future__ import annotations

import math
import time

import pytest

from robotsnap import topics
from robotsnap.bridge import codec
from robotsnap.bridge.server import RobotSNAPBridge
from robotsnap_ros2 import Ros2Gateway
from unity_peer import UnityPeer, unity_laserscan, unity_occupancy_grid, unity_odometry

rclpy = pytest.importorskip("rclpy", reason="no ROS2 install is sourced on this interpreter")
TFMessage = pytest.importorskip("tf2_msgs.msg").TFMessage
qos = pytest.importorskip("rclpy.qos", reason="no ROS2 QoS module on this interpreter")

_DEADLINE = 20.0


def _wait_for(predicate, node=None, timeout=_DEADLINE):
    """Poll while spinning ``node``, so a graph match can happen under the wait."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if node is not None:
            rclpy.spin_once(node, timeout_sec=0.02)
        value = predicate()
        if value:
            return value
        time.sleep(0.02)
    return None


def _entries(messages):
    """Every ``TransformStamped`` of every message received, in arrival order."""
    return [entry for message in messages for entry in message.transforms]


def _link(messages, parent, child):
    """The link ``parent -> child`` one of the messages carried, or None."""
    for entry in _entries(messages):
        if entry.header.frame_id == parent and entry.child_frame_id == child:
            return entry
    return None


@pytest.fixture
def wired():
    """A bridge with a Unity peer on it and a mirror beside it, on the graph this process is on."""
    bridge = RobotSNAPBridge(host="127.0.0.1", port=0)
    bridge.start()
    peer = UnityPeer(bridge.port)
    peer.recv_handshake()
    peer.register_publisher(topics.ODOM, "nav_msgs/Odometry")
    peer.register_publisher(topics.SCAN, "sensor_msgs/LaserScan")
    peer.register_publisher(topics.MAP, "nav_msgs/OccupancyGrid")

    # The registrations are frames on the socket, read by the bridge's own thread: the mirror is
    # only meaningful once they have landed.
    assert _wait_for(
        lambda: all(
            topics.base(topic) in bridge.announced_topics()
            for topic in (topics.ODOM, topics.SCAN, topics.MAP)
        ),
        timeout=5.0,
    ), "the peer's registrations never reached the bridge"

    gateway = Ros2Gateway(bridge, node_name="robotsnap_test_tf_gateway")
    gateway.start()
    node = rclpy.create_node("robotsnap_test_tf_graph")
    try:
        yield bridge, peer, gateway, node
    finally:
        node.destroy_node()
        gateway.stop()
        peer.close()
        bridge.stop()


def test_the_session_streams_become_a_transform_tree(wired):
    bridge, peer, gateway, node = wired
    dynamic: list = []
    static: list = []
    node.create_subscription(TFMessage, "/tf", dynamic.append, 10)
    # The static topic is latched: a subscriber that joins after the tree crossed still hears it, so
    # the test subscribes with the profile the publisher uses.
    latching = qos.QoSProfile(
        depth=1,
        reliability=qos.ReliabilityPolicy.RELIABLE,
        durability=qos.DurabilityPolicy.TRANSIENT_LOCAL,
    )
    node.create_subscription(TFMessage, "/tf_static", static.append, latching)

    odom = unity_odometry(x=2.5, y=-1.25, yaw=0.5, frame_id="/odom", stamp=1.0)
    scan = unity_laserscan([1.0, 2.0], frame_id="/laser", stamp=1.0)
    grid = unity_occupancy_grid([[0, 0], [0, 0]], frame_id="/map", stamp=1.0)

    # The mirror creates its publisher on the first message the session sends, and DDS matches the new
    # publisher asynchronously, so the session repeats until the graph has paired them - which is what
    # a real session does anyway, at its own rate.
    deadline = time.monotonic() + _DEADLINE
    while time.monotonic() < deadline:
        peer.publish(topics.ODOM, codec.encode(topics.ODOMETRY_TYPE, odom))
        peer.publish(topics.SCAN, codec.encode(topics.LASER_SCAN_TYPE, scan))
        peer.publish(topics.MAP, codec.encode(topics.OCCUPANCY_GRID_TYPE, grid))
        rclpy.spin_once(node, timeout_sec=0.05)
        if _link(dynamic, "odom", "base_link") and _link(static, "map", "odom"):
            break

    link = _link(dynamic, "odom", "base_link")
    assert link is not None, "no odom -> base_link transform reached the graph"
    assert link.transform.translation.x == pytest.approx(2.5)
    assert link.transform.translation.y == pytest.approx(-1.25)
    rotation = link.transform.rotation
    assert math.atan2(2.0 * rotation.w * rotation.z, 1.0 - 2.0 * rotation.z**2) == pytest.approx(0.5)

    for parent, child in (("map", "odom"), ("base_link", "laser")):
        entry = _link(static, parent, child)
        assert entry is not None, f"no {parent} -> {child} transform reached the graph"
        assert entry.transform.translation.x == pytest.approx(0.0)
        assert entry.transform.rotation.w == pytest.approx(1.0)

    # tf2 refuses a frame id with a leading slash, and the session spells them with one.
    for entry in _entries(dynamic) + _entries(static):
        assert not entry.header.frame_id.startswith("/")
        assert not entry.child_frame_id.startswith("/")

    assert gateway.stats()["tf"] > 0
    assert gateway.stats()["tf_static"] > 0
    assert gateway.last_error is None


def test_the_static_tree_reaches_a_reader_that_joins_late(wired):
    """A reader that arrives after the map did still gets ``/tf_static``.

    That is the whole point of the latched profile: RViz and a navigation stack are started after
    the session, and a static link heard only by whoever was already listening when it crossed would
    never reach them. The publisher has to be built from the real ``rclpy.qos`` module for that,
    which ``import rclpy`` alone does not import.
    """
    bridge, peer, gateway, node = wired
    grid = unity_occupancy_grid([[0, 0], [0, 0]], frame_id="/map", stamp=1.0)
    odom = unity_odometry(x=1.0, y=0.0, yaw=0.0, frame_id="/odom", stamp=1.0)

    deadline = time.monotonic() + _DEADLINE
    while time.monotonic() < deadline and gateway.stats()["tf_static"] == 0:
        peer.publish(topics.MAP, codec.encode(topics.OCCUPANCY_GRID_TYPE, grid))
        peer.publish(topics.ODOM, codec.encode(topics.ODOMETRY_TYPE, odom))
        rclpy.spin_once(node, timeout_sec=0.05)
    assert gateway.stats()["tf_static"] > 0, "the gateway never published a static tree"

    # Only now does the reader arrive, with the profile a latched topic is read with.
    late: list = []
    latching = qos.QoSProfile(
        depth=1,
        reliability=qos.ReliabilityPolicy.RELIABLE,
        durability=qos.DurabilityPolicy.TRANSIENT_LOCAL,
    )
    node.create_subscription(TFMessage, "/tf_static", late.append, latching)

    assert _wait_for(lambda: _link(late, "map", "odom"), node=node) is not None, (
        "/tf_static did not reach a reader that joined after it was published"
    )
