"""The mirror between a Unity session and a ROS2 graph, without a ROS2 install.

The gateway's job is a translation of placement rather than of content: it takes what arrived on the
socket and puts it on the graph, and takes what arrives on the graph and writes it back on the
socket. Everything it needs from ``rclpy`` is small enough to stand in for, so the two directions,
the skip bookkeeping and the shutdown are all asserted here on a fake support, on any machine. The
live graph is exercised separately, in ``test_ros2_gateway_live.py``.
"""

from __future__ import annotations

import pytest

from robotsnap import topics
from robotsnap_ros2 import Ros2Gateway, Ros2Unavailable, unavailable_reason


class _FakePublisher:
    def __init__(self, message_type, topic):
        self.message_type = message_type
        self.topic = topic
        self.published: list[object] = []

    def publish(self, message):
        self.published.append(message)


class _FakeSubscription:
    def __init__(self, message_type, topic, callback):
        self.message_type = message_type
        self.topic = topic
        self.callback = callback


class _FakeNode:
    def __init__(self, name):
        self.name = name
        self.publishers: dict[tuple, _FakePublisher] = {}
        self.subscriptions: list[_FakeSubscription] = []
        self.destroyed = False
        self.offered: list[tuple[str, list[str]]] = []

    def create_publisher(self, message_type, topic, qos):
        publisher = _FakePublisher(message_type, topic)
        self.publishers[(message_type, topic)] = publisher
        return publisher

    def create_subscription(self, message_type, topic, callback, qos):
        subscription = _FakeSubscription(message_type, topic, callback)
        self.subscriptions.append(subscription)
        return subscription

    def get_topic_names_and_types(self):
        return list(self.offered)

    def destroy_node(self):
        self.destroyed = True

    def publisher_for(self, topic):
        for publisher in self.publishers.values():
            if publisher.topic == topic:
                return publisher
        return None


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


class _FakeExecutor:
    def __init__(self):
        self.nodes: list[_FakeNode] = []
        self.shut_down = False

    def add_node(self, node):
        self.nodes.append(node)

    def remove_node(self, node):
        if node in self.nodes:
            self.nodes.remove(node)

    def spin_once(self, timeout_sec=0.0):
        return None

    def shutdown(self):
        self.shut_down = True


class _FakeSupport:
    """What :class:`Ros2Support` answers, with the message classes replaced by their names."""

    def __init__(self, known=None, unknown=()):
        self.rclpy = _FakeRclpy()
        self.rclpy.init()
        self._known = dict(known or {topics.ODOMETRY_TYPE: "Odometry", topics.TWIST_TYPE: "Twist"})
        self._unknown = set(unknown)
        self.serialized: list[object] = []
        self.deserialized: list[bytes] = []
        self.executor = _FakeExecutor()

    def get_message(self, name):
        key = str(name).strip()
        if key in self._unknown:
            raise Ros2Unavailable(f"the graph has no message type {key}")
        return self._known.get(key, key)

    def serialize(self, message):
        self.serialized.append(message)
        return b"graph:" + bytes(str(message), "utf-8")

    def deserialize(self, payload, message_type):
        self.deserialized.append(bytes(payload))
        return {"type": message_type, "payload": bytes(payload)}

    def new_executor(self):
        return self.executor


class _FakeBridge:
    """The slice of :class:`RobotSNAPBridge` a mirror uses."""

    def __init__(self, *, published=None, subscriptions=None, names=None):
        self._listeners: list = []
        self._published = dict(published or {})
        self._subscriptions = dict(subscriptions or {})
        self._names = dict(names or {})
        self.sent: list[tuple[str, bytes]] = []

    def add_message_listener(self, listener):
        self._listeners.append(listener)

    def remove_message_listener(self, listener):
        if listener in self._listeners:
            self._listeners.remove(listener)

    def published_topics(self):
        return dict(self._published)

    def subscriptions(self):
        return dict(self._subscriptions)

    def topic_types(self):
        # The real bridge answers from its seeded table as well as the peer's announcements, which is
        # what lets a command topic be subscribed before Unity has said anything about it. An
        # announcement without a type does not overwrite the seed, exactly as on the real bridge.
        known = topics.base_types()
        known.update({key: value for key, value in self._published.items() if value})
        return known

    def destination_for(self, topic):
        key = topics.base(topic)
        return self._names.get(key, "/" + key)

    def send_message(self, topic, payload):
        self.sent.append((topic, bytes(payload)))
        return True

    # -- what the session would do -----------------------------------------

    def emit(self, topic, payload=b"raw", msg_type=topics.ODOMETRY_TYPE):
        key = topics.base(topic)
        self._published[key] = msg_type
        for listener in list(self._listeners):
            listener(key, payload, msg_type)


@pytest.fixture
def mirror(monkeypatch):
    """A gateway over a fake support, on a bridge that can be made to speak."""
    support = _FakeSupport()
    monkeypatch.setattr(
        "robotsnap_ros2.gateway.ros2_support", lambda: support
    )
    bridge = _FakeBridge(
        published={topics.base(topics.ODOM): topics.ODOMETRY_TYPE},
        subscriptions={topics.base(topics.CMD_VEL): topics.CMD_VEL},
    )
    gateway = Ros2Gateway(bridge, node_name="test_mirror")
    yield gateway, bridge, support
    gateway.stop()


def test_a_session_message_reaches_the_graph(mirror):
    gateway, bridge, support = mirror

    bridge.emit(topics.ODOM, b"odom-bytes")

    node = support.rclpy.nodes[0]
    publisher = node.publisher_for(topics.ODOM)
    assert publisher is not None, "the mirror opens the topic the session announced"
    assert len(publisher.published) == 1
    assert publisher.published[0] == {"type": "Odometry", "payload": b"odom-bytes"}
    assert support.deserialized == [b"odom-bytes"], "the payload crosses as it arrived"
    assert gateway.stats()["published"] == 1


def test_the_graph_drives_the_session_on_the_name_unity_registered(mirror):
    gateway, bridge, support = mirror
    bridge._names[topics.base(topics.CMD_VEL)] = "/robot_1/cmd_vel"

    gateway._discover()
    node = support.rclpy.nodes[0]
    assert [sub.topic for sub in node.subscriptions] == [topics.CMD_VEL]

    node.subscriptions[0].callback("a-twist")

    assert bridge.sent == [("/robot_1/cmd_vel", b"graph:a-twist")]
    assert gateway.stats()["received"] == 1


def test_a_type_the_graph_cannot_carry_is_skipped_once(mirror):
    gateway, bridge, support = mirror
    support._unknown.add(topics.ODOMETRY_TYPE)

    bridge.emit(topics.ODOM, b"first")
    bridge.emit(topics.ODOM, b"second")

    node = support.rclpy.nodes[0]
    assert node.publisher_for(topics.ODOM) is None
    refused = f"{topics.base(topics.ODOM)} (publishing)"
    assert list(gateway.skipped()) == [refused]
    assert "no message type" in gateway.skipped()[refused]
    assert gateway.stats()["skipped"] == 1, "the reason is said once, not per message"
    assert gateway.last_error is None, "a topic the graph cannot carry is not an error"


def test_a_topic_unity_only_listens_to_is_not_offered_on_the_graph(mirror):
    gateway, bridge, support = mirror

    gateway._discover()

    node = support.rclpy.nodes[0]
    assert node.publisher_for(topics.CMD_VEL) is None, (
        "Unity subscribes to /cmd_vel; a publisher there would announce a stream it never sends"
    )
    assert "/cmd_vel" in [sub.topic for sub in node.subscriptions]


def test_a_refusal_to_publish_does_not_silence_the_same_name_the_other_way(mirror):
    gateway, bridge, support = mirror
    bridge._published[topics.base(topics.ODOM)] = ""
    bridge._subscriptions[topics.base(topics.ODOM)] = topics.ODOM

    gateway._discover()

    node = support.rclpy.nodes[0]
    assert node.publisher_for(topics.ODOM) is None
    assert topics.base(topics.ODOM) + " (publishing)" in gateway.skipped()
    # The session named no type for publishing, which is one refusal; the same name is still one the
    # graph can carry the other way, and a shared memo would have dropped the subscription with it.
    assert [sub.topic for sub in node.subscriptions if sub.topic == topics.ODOM] == [topics.ODOM]


def test_a_namespaced_command_the_graph_offers_is_picked_up(mirror):
    gateway, bridge, support = mirror
    node = support.rclpy.nodes[0]
    node.offered = [("/robot_2/cmd_vel", [topics.TWIST_TYPE])]

    gateway._discover()

    subscribed = [sub.topic for sub in node.subscriptions]
    assert "/robot_2/cmd_vel" in subscribed, (
        "a namespaced command the graph already offers is subscribed to as well"
    )
    fleet = next(sub for sub in node.subscriptions if sub.topic == "/robot_2/cmd_vel")
    fleet.callback("fleet-twist")
    assert bridge.sent == [("/robot_2/cmd_vel", b"graph:fleet-twist")], (
        "a command nobody registered goes to the name the graph used"
    )


def test_stop_removes_the_listener_and_is_idempotent(mirror):
    gateway, bridge, support = mirror
    assert bridge._listeners, "the gateway listens while it is open"

    gateway.stop()
    gateway.stop()

    assert bridge._listeners == []
    assert support.rclpy.nodes[0].destroyed
    assert gateway.closed


def test_the_mirror_says_why_it_cannot_run_without_ros2(monkeypatch):
    def refuse():
        raise Ros2Unavailable("no rclpy on this interpreter")

    monkeypatch.setattr("robotsnap_ros2.gateway.ros2_support", refuse)
    assert unavailable_reason() == "no rclpy on this interpreter"
