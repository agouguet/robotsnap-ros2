"""The ROS2 session as a transport the RL side can drive, without a ROS2 install.

The claim this file exists for is the one the architecture turns on: a run asked for ``ros2`` must
reach the same façade, the same commands and the same ``cmd_vel`` as a run on the socket, and the
only thing that changes is where the bytes go. ``Ros2Client`` is stood up on a fake ``rclpy`` and a
fake bridge surface, then driven through ``RobotSNAPClient`` itself, so what is asserted is the
client the environments use rather than the ROS2 wrapper alone.

The real graph is exercised separately, in ``test_ros2_write_live.py``.
"""

from __future__ import annotations

import json
import time

import pytest

from robotsnap import topics
from robotsnap.bridge import codec
from robotsnap.client import RobotSNAPClient, open_client
from robotsnap_ros2 import Ros2Unavailable

_TWIST = codec.TYPESTORE.types["geometry_msgs/msg/Twist"]


class _FakePublisher:
    """Records what was published, and hands it to the node's own answer hook if it has one."""

    def __init__(self, message_type, topic, node=None):
        self.message_type = message_type
        self.topic = topic
        self.published: list[bytes] = []
        self._node = node

    def publish(self, message):
        self.published.append(bytes(message))
        hook = getattr(self._node, "on_publish", None)
        if hook is not None:
            hook(self.topic, message)


class _FakeNode:
    def __init__(self, name):
        self.name = name
        self.publishers: dict[str, _FakePublisher] = {}
        self.subscriptions: list[tuple] = []
        self.subscriber_counts: dict[str, int] = {}
        self.destroyed = False
        self.offered: list[tuple[str, list[str]]] = []
        #: Called with ``(topic, message)`` for every publish, which is how a test stands in for the
        #: simulator answering a command.
        self.on_publish = None

    def create_publisher(self, message_type, topic, qos):
        publisher = _FakePublisher(message_type, topic, self)
        self.publishers.setdefault(topic, publisher)
        return publisher

    def create_subscription(self, message_type, topic, callback, qos):
        self.subscriptions.append((topic, message_type, callback))
        return (topic, message_type, callback)

    def count_subscribers(self, topic):
        return self.subscriber_counts.get(topic, 0)

    def get_topic_names_and_types(self):
        return list(self.offered)

    def destroy_node(self):
        self.destroyed = True


class _FakeRclpy:
    def __init__(self):
        self.nodes: list[_FakeNode] = []
        self.initialized = False
        self.shut_down = False

    def ok(self):
        return self.initialized and not self.shut_down

    def init(self, args=None):
        self.initialized = True

    def create_node(self, name):
        node = _FakeNode(name)
        self.nodes.append(node)
        return node

    def spin_once(self, node, timeout_sec=0.0):
        return None

    def shutdown(self):
        self.shut_down = True


class _FakeSupport:
    """The ROS2 pieces a session uses, with CDR standing in for the typed message."""

    def __init__(self, rclpy):
        self.rclpy = rclpy

    def get_message(self, name):
        return str(name)

    def serialize(self, message):
        return bytes(message)

    def deserialize(self, payload, message_type):
        return bytes(payload)


@pytest.fixture
def session(monkeypatch):
    """A ROS2 client on fake plumbing, and the node it is talking to."""
    rclpy = _FakeRclpy()
    monkeypatch.setattr("robotsnap_ros2.support.import_rclpy", lambda: (rclpy, {
        "OccupancyGrid": "OccupancyGrid",
        "Odometry": "Odometry",
        "LaserScan": "LaserScan",
        "String": "String",
    }))
    monkeypatch.setattr("robotsnap_ros2.support.Ros2Support", lambda rclpy, types=None: _FakeSupport(rclpy))

    from robotsnap_ros2 import Ros2Client

    client = Ros2Client(node_name="test_session")
    node = rclpy.nodes[0]
    try:
        yield client, node
    finally:
        client.stop()


def _state_message(playing=True, held=False):
    return type("String", (), {"data": json.dumps({"playing": playing, "held": held})})()


def test_a_cmd_vel_sent_over_the_client_reaches_the_graph(session):
    client, node = session
    client._stored[topics.base(topics.SIMULATION_STATE)] = _state_message()
    client._received_at[topics.base(topics.SIMULATION_STATE)] = time.monotonic()
    node.subscriber_counts[topics.SIMULATION_CONTROL] = 1

    façade = RobotSNAPClient(bridge=client.bridge)
    try:
        assert façade.wait_until_ready(timeout=1.0), façade.last_error
        assert façade.send_cmd_vel(0.6, -0.4)
    finally:
        façade.stop()

    publisher = node.publishers[topics.CMD_VEL]
    assert len(publisher.published) == 1
    twist = codec.decode(topics.TWIST_TYPE, publisher.published[0])
    assert twist.linear.x == pytest.approx(0.6)
    assert twist.angular.z == pytest.approx(-0.4)


def test_a_command_addressed_to_one_robot_uses_its_own_name(session):
    client, node = session

    assert client.send_cmd_vel(0.2, 0.1, robot="robot_2")

    assert topics.robot_topic("robot_2", topics.CMD_VEL) in node.publishers
    assert topics.CMD_VEL not in node.publishers, "the primary robot was not addressed"


def test_a_control_command_is_published_as_the_string_the_endpoint_carries(session):
    client, node = session

    assert client.publish(
        topics.SIMULATION_CONTROL, topics.STRING_TYPE, _string_message({"command": "play"})
    )

    publisher = node.publishers[topics.SIMULATION_CONTROL]
    body = codec.decode(topics.STRING_TYPE, publisher.published[0]).data
    assert json.loads(body) == {"command": "play"}


def test_the_facade_reads_the_acknowledgement_the_graph_carries(session):
    client, node = session
    node.subscriber_counts[topics.SIMULATION_CONTROL] = 1
    client._stored[topics.base(topics.SIMULATION_STATE)] = _state_message()
    client._received_at[topics.base(topics.SIMULATION_STATE)] = time.monotonic()

    # The endpoint carries the simulator's answer back on the result topic once the command has left,
    # which the client waits for by watching that topic's count - the same poll it uses on a socket.
    def answer(topic, message):
        if topic == topics.SIMULATION_CONTROL:
            client._store(
                topics.base(topics.SIMULATION_CONTROL_RESULT),
                _json_string({"ok": True, "command": "play"}),
            )

    node.on_publish = answer

    façade = RobotSNAPClient(bridge=client.bridge)
    try:
        assert façade.wait_until_ready(timeout=1.0), façade.last_error
        result = façade.play()
        assert result is not None and result.get("ok") is True
    finally:
        façade.stop()


def test_waiting_for_a_subscription_is_the_graph_seeing_unity_listen(session):
    client, node = session
    bridge = client.bridge

    assert bridge.wait_for_subscription(topics.SIMULATION_CONTROL, timeout=0.05) is False
    node.subscriber_counts[topics.SIMULATION_CONTROL] = 1
    assert bridge.wait_for_subscription(topics.SIMULATION_CONTROL, timeout=0.05) is True


def test_the_facade_owns_the_lifetime_of_the_session_it_started(session):
    client, node = session
    bridge = client.bridge
    assert bridge.is_running is False

    façade = RobotSNAPClient(bridge=bridge)
    assert bridge.is_running is True

    façade.stop()
    assert bridge.is_running is False
    assert node.destroyed, "the node the façade started is the node it closes"


def test_open_client_refuses_a_transport_it_cannot_honour(monkeypatch):
    with pytest.raises(ValueError):
        open_client(transport="carrier-pigeon")

    def refuse():
        raise Ros2Unavailable("no rclpy on this interpreter")

    monkeypatch.setattr("robotsnap_ros2.support.import_rclpy", refuse)
    with pytest.raises(Ros2Unavailable):
        open_client(transport="ros2")


def _string_message(body):
    return codec.TYPESTORE.types["std_msgs/msg/String"](data=json.dumps(body))


def _json_string(body):
    """What a ``std_msgs/String`` looks like once the graph has handed it over."""
    return type("String", (), {"data": json.dumps(body)})()
