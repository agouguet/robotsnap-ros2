"""The mirror on a real ROS2 graph, with a fake Unity on the socket.

The unit suite stands in for ``rclpy`` and proves the placement; this one proves the CDR, the
discovery and the topic names against a real graph, which is the only place a wrong message type or a
wrong spelling would show. It is skipped when no ROS2 install is sourced, for the same reason
``test_ros2.py`` is: a machine either has one or does not.

    source /opt/ros/humble/setup.bash
    python -m pytest tests/test_ros2_gateway_live.py -q

With ROS2 sourced, ``pytest``'s own entry points can break collection on some installs; run it with
``PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`` if that happens.
"""

from __future__ import annotations

import math
import time

import pytest

from robotsnap import topics
from robotsnap.bridge import codec, protocol
from robotsnap.bridge.server import RobotSNAPBridge
from robotsnap_ros2 import Ros2Gateway
from unity_peer import UnityPeer

rclpy = pytest.importorskip("rclpy", reason="no ROS2 install is sourced on this interpreter")
Odometry = pytest.importorskip("nav_msgs.msg").Odometry
Twist = pytest.importorskip("geometry_msgs.msg").Twist

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


@pytest.fixture
def wired():
    """A bridge with a Unity peer on it and a mirror beside it, on the graph this process is on."""
    bridge = RobotSNAPBridge(host="127.0.0.1", port=0)
    bridge.start()
    peer = UnityPeer(bridge.port)
    peer.recv_handshake()
    peer.register_publisher(topics.ODOM, "nav_msgs/Odometry")
    peer.register_subscriber(topics.CMD_VEL, "geometry_msgs/Twist")

    # The registrations are frames on the socket, read by the bridge's own thread: the mirror is
    # only meaningful once they have landed.
    assert _wait_for(
        lambda: topics.base(topics.ODOM) in bridge.announced_topics()
        and topics.base(topics.CMD_VEL) in bridge.subscriptions(),
        timeout=5.0,
    ), "the peer's registrations never reached the bridge"

    gateway = Ros2Gateway(bridge, node_name="robotsnap_test_gateway")
    gateway.start()
    node = rclpy.create_node("robotsnap_test_graph")
    try:
        yield bridge, peer, gateway, node
    finally:
        node.destroy_node()
        gateway.stop()
        peer.close()
        bridge.stop()


def test_unity_odometry_reaches_a_graph_subscriber(wired):
    bridge, peer, gateway, node = wired
    received: list = []
    node.create_subscription(Odometry, topics.ODOM, received.append, 10)

    # The mirror creates its publisher on the first message the session sends, and DDS matches the
    # new publisher asynchronously, so the session repeats until the graph has paired them. This is
    # what a real session does anyway: it publishes at its own rate.
    deadline = time.monotonic() + _DEADLINE
    while not received and time.monotonic() < deadline:
        peer.publish_odom(x=2.5, y=-1.25, yaw=0.5, stamp=1.0)
        rclpy.spin_once(node, timeout_sec=0.05)

    assert received, "the odometry never reached the graph"
    # The graph may carry another node's odometry as well, so the test looks for its own message
    # rather than reading the last one to arrive.
    mine = [
        message
        for message in received
        if message.pose.pose.position.x == pytest.approx(2.5)
    ]
    assert mine, "the odometry this session published never reached the graph"
    message = mine[-1]
    assert message.pose.pose.position.y == pytest.approx(-1.25)
    assert message.twist.twist.linear.x == pytest.approx(0.0)
    assert math.atan2(
        2.0 * (message.pose.pose.orientation.w * message.pose.pose.orientation.z),
        1.0 - 2.0 * message.pose.pose.orientation.z**2,
    ) == pytest.approx(0.5)
    assert gateway.stats()["published"] > 0


def test_a_graph_command_reaches_unity(wired):
    bridge, peer, gateway, node = wired
    publisher = node.create_publisher(Twist, topics.CMD_VEL, 10)
    assert _wait_for(lambda: publisher.get_subscription_count() > 0, node=node), (
        "the mirror never subscribed to the command topic"
    )

    twist = Twist()
    twist.linear.x = 0.6
    twist.angular.z = -0.4
    publisher.publish(twist)

    # A graph is shared: another node may be publishing its own `/cmd_vel` beside this one, and the
    # mirror is right to forward that too. The test waits for *its* command to come through rather
    # than for the next frame, which would be whoever published first.
    peer.sock.settimeout(1.0)
    deadline = time.monotonic() + _DEADLINE
    while time.monotonic() < deadline:
        try:
            destination, payload = protocol.read_frame(peer.sock)
        except (TimeoutError, OSError):
            continue
        assert destination == topics.CMD_VEL
        message = codec.decode(topics.TWIST_TYPE, payload)
        if message.linear.x == pytest.approx(0.6) and message.angular.z == pytest.approx(-0.4):
            return
    pytest.fail("the command published on the graph never reached Unity")


def test_the_mirror_does_not_publish_the_streams_unity_only_listens_to(wired):
    bridge, peer, gateway, node = wired
    gateway._discover()
    rclpy.spin_once(node, timeout_sec=0.2)

    publishers = [info.node_name for info in node.get_publishers_info_by_topic(topics.CMD_VEL)]
    assert "robotsnap_test_gateway" not in publishers, (
        "a publisher on /cmd_vel would tell a navigation node that Unity authors the command"
    )
