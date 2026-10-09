"""The transform tree a session implies, composed without a ROS2 install.

Most of the gateway's job is a translation of placement: what arrived on the socket is put on the
graph. One part is an addition - the ``/tf`` tree the session's streams imply and never publish. That
composition is plain arithmetic on decoded messages, so it is asserted here on the ``rosbags``
classes the core already builds, with the graph's ``rclpy`` stood in for. The live graph is
exercised separately, in ``test_ros2_tf_live.py``.
"""

from __future__ import annotations

import math
from types import SimpleNamespace

import pytest

from robotsnap import topics
from robotsnap.bridge import codec
from robotsnap_ros2 import Ros2Gateway
from robotsnap_ros2.tf import (
    TF_MESSAGE_TYPE,
    TfComposer,
    Transform,
    build_tf_message,
    normalise_frame,
)
from unity_peer import unity_laserscan, unity_occupancy_grid, unity_odometry


class _RecordingPublisher:
    def __init__(self, message_type, topic, qos):
        self.message_type = message_type
        self.topic = topic
        self.qos = qos
        self.published: list[object] = []

    def publish(self, message):
        self.published.append(message)


class _FakeNode:
    def __init__(self, name):
        self.name = name
        #: Keyed by topic, so a test reads the publisher of ``/tf`` by name; a topic is made once.
        self.publishers: dict[str, _RecordingPublisher] = {}
        self.subscriptions: list = []
        self.destroyed = False

    def create_publisher(self, message_type, topic, qos):
        publisher = _RecordingPublisher(message_type, topic, qos)
        self.publishers[topic] = publisher
        return publisher

    def create_subscription(self, message_type, topic, callback, qos):
        subscription = object()
        self.subscriptions.append(subscription)
        return subscription

    def get_topic_names_and_types(self):
        return []

    def destroy_node(self):
        self.destroyed = True


class _FakeQos:
    """The slice of ``rclpy.qos`` the static publisher is built from, kept readable for a test."""

    class ReliabilityPolicy:
        RELIABLE = "reliable"

    class DurabilityPolicy:
        TRANSIENT_LOCAL = "transient_local"

    @staticmethod
    def QoSProfile(depth, reliability=None, durability=None):
        return {"depth": depth, "reliability": reliability, "durability": durability}


class _FakeRclpy:
    def __init__(self):
        self.qos = _FakeQos()
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
    """What :class:`Ros2Support` answers, over the ``rosbags`` typestore the core already holds.

    Resolving a message name to its real ``rosbags`` class is what lets the composition and the
    message building be run here for real rather than against a stand-in that could only agree with
    itself: only the node and the QoS profile are faked.
    """

    def __init__(self):
        self.rclpy = _FakeRclpy()
        self.rclpy.init()
        self.executor = _FakeExecutor()
        self.types = codec.TYPESTORE.types

    def get_message(self, name):
        try:
            return self.types[str(name).strip()]
        except KeyError as exc:
            raise AssertionError(f"unknown message type {name}") from exc

    def deserialize(self, payload, message_type):
        return codec.decode(message_type.__msgtype__, payload)

    def serialize(self, message):
        return codec.encode(message.__msgtype__, message)

    def new_executor(self):
        return self.executor


class _FakeBridge:
    """The slice of :class:`RobotSNAPBridge` a mirror and the transform synthesis read."""

    def __init__(self, *, published=None, subscriptions=None):
        self._listeners: list = []
        self._published = dict(published or {})
        self._subscriptions = dict(subscriptions or {})

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
        return topics.base_types()

    def destination_for(self, topic):
        return "/" + topics.base(topic)

    def send_message(self, topic, payload):
        return True

    def emit(self, topic, message):
        """Publish one decoded message under ``topic``, as the socket's read thread would."""
        key = topics.base(topic)
        msg_type = self._published.get(key) or topics.base_types().get(key, "")
        payload = codec.encode(message.__msgtype__, message)
        for listener in list(self._listeners):
            listener(key, payload, msg_type)


@pytest.fixture
def mirror(monkeypatch):
    """A gateway over a fake support, on a bridge announcing the three streams it decodes."""
    support = _FakeSupport()
    monkeypatch.setattr("robotsnap_ros2.gateway.ros2_support", lambda: support)
    bridge = _FakeBridge(
        published={
            topics.base(topics.ODOM): topics.ODOMETRY_TYPE,
            topics.base(topics.SCAN): topics.LASER_SCAN_TYPE,
            topics.base(topics.MAP): topics.OCCUPANCY_GRID_TYPE,
        }
    )
    gateway = Ros2Gateway(bridge, node_name="test_tf")
    yield gateway, bridge, support
    gateway.stop()


def _fleet_odometry(robot: str = "robot_2", **kwargs):
    """An odometry whose frames are a fleet robot's own, the way the namespaced stream spells them."""
    message = unity_odometry(frame_id=f"/{robot}/odom", **kwargs)
    message.child_frame_id = f"{robot}/base_link"
    return message


# -- the composition ------------------------------------------------------


def test_a_frame_loses_its_leading_slash_and_its_spaces():
    assert normalise_frame("/map") == "map"
    assert normalise_frame("  /robot_2/laser ") == "robot_2/laser"
    assert normalise_frame("base_link") == "base_link"
    assert normalise_frame("/") == "", "an empty frame stays empty rather than becoming a name"
    assert normalise_frame(None) == ""


def test_odometry_becomes_one_dynamic_transform():
    composer = TfComposer()
    composer.observe(
        topics.base(topics.ODOM),
        topics.ODOMETRY_TYPE,
        unity_odometry(x=2.5, y=-1.25, yaw=math.pi / 2, frame_id="/odom", stamp=1.25),
    )

    links = composer.dynamic_transforms()
    assert len(links) == 1
    link = links[0]
    assert (link.parent, link.child) == ("odom", "base_link")
    assert link.translation == pytest.approx((2.5, -1.25, 0.0))
    assert link.rotation == pytest.approx(
        (0.0, 0.0, math.sin(math.pi / 4), math.cos(math.pi / 4))
    )
    assert link.stamp == (1, 250_000_000), "the transform is stamped with the odometry's header"


def test_the_static_tree_is_the_identity_of_the_frames_it_joins():
    composer = TfComposer()
    composer.observe("odom", topics.ODOMETRY_TYPE, unity_odometry(frame_id="/odom", stamp=1.0))
    composer.observe(
        "scan", topics.LASER_SCAN_TYPE, unity_laserscan([1.0], frame_id="/laser", stamp=1.0)
    )
    composer.observe(
        "map",
        topics.OCCUPANCY_GRID_TYPE,
        unity_occupancy_grid([[0, 0], [0, 0]], frame_id="/map", stamp=1.0),
    )

    assert composer.static_transforms() == (
        Transform("map", "odom"),
        Transform("base_link", "laser"),
    )


def test_a_second_robot_gets_its_own_tree():
    composer = TfComposer()
    composer.observe("odom", topics.ODOMETRY_TYPE, unity_odometry(x=1.0, frame_id="/odom"))
    composer.observe("scan", topics.LASER_SCAN_TYPE, unity_laserscan([1.0], frame_id="/laser"))
    composer.observe(
        "map", topics.OCCUPANCY_GRID_TYPE, unity_occupancy_grid([[0]], frame_id="/map")
    )
    composer.observe("robot_2/odom", topics.ODOMETRY_TYPE, _fleet_odometry(x=3.0, y=4.0))
    composer.observe(
        "robot_2/scan",
        topics.LASER_SCAN_TYPE,
        unity_laserscan([1.0], frame_id="robot_2/laser"),
    )

    dynamic = composer.dynamic_transforms()
    assert [(link.parent, link.child) for link in dynamic] == [
        ("odom", "base_link"),
        ("robot_2/odom", "robot_2/base_link"),
    ]
    assert composer.static_transforms() == (
        Transform("map", "odom"),
        Transform("base_link", "laser"),
        Transform("map", "robot_2/odom"),
        Transform("robot_2/base_link", "robot_2/laser"),
    ), "the fleet robot hangs off the session's map, not the primary robot's odom"


def test_no_scan_means_no_lidar_link():
    composer = TfComposer()
    composer.observe("odom", topics.ODOMETRY_TYPE, unity_odometry(frame_id="/odom"))
    composer.observe(
        "map", topics.OCCUPANCY_GRID_TYPE, unity_occupancy_grid([[0]], frame_id="/map")
    )

    assert composer.static_transforms() == (Transform("map", "odom"),)


def test_an_odometry_without_a_pose_yields_no_transform():
    composer = TfComposer()
    header = SimpleNamespace(frame_id="/odom", stamp=SimpleNamespace(sec=1, nanosec=0))
    composer.observe(
        "odom",
        topics.ODOMETRY_TYPE,
        SimpleNamespace(header=header, child_frame_id="base_link", pose=None),
    )
    composer.observe("scan", topics.LASER_SCAN_TYPE, unity_laserscan([1.0], frame_id="/laser"))
    composer.observe(
        "map", topics.OCCUPANCY_GRID_TYPE, unity_occupancy_grid([[0]], frame_id="/map")
    )

    assert composer.dynamic_transforms() == (), "a pose is never invented"
    assert composer.static_transforms() == (
        Transform("map", "odom"),
        Transform("base_link", "laser"),
    ), "the frames the message named are kept, even without a pose"


def test_the_static_tree_is_handed_over_only_when_it_changes():
    composer = TfComposer()
    composer.observe("odom", topics.ODOMETRY_TYPE, unity_odometry(frame_id="/odom"))
    composer.observe("scan", topics.LASER_SCAN_TYPE, unity_laserscan([1.0], frame_id="/laser"))
    composer.observe(
        "map", topics.OCCUPANCY_GRID_TYPE, unity_occupancy_grid([[0]], frame_id="/map")
    )

    first = composer.new_static_transforms()
    assert first == composer.static_transforms()
    assert composer.new_static_transforms() == (), "the same tree is not published twice"
    composer.observe(
        "scan", topics.LASER_SCAN_TYPE, unity_laserscan([2.0], frame_id="/laser", stamp=9.0)
    )
    assert composer.new_static_transforms() == (), "a newer scan of the same frame changes nothing"

    composer.observe("robot_2/odom", topics.ODOMETRY_TYPE, _fleet_odometry())
    composer.observe(
        "robot_2/scan",
        topics.LASER_SCAN_TYPE,
        unity_laserscan([1.0], frame_id="robot_2/laser"),
    )
    grown = composer.new_static_transforms()
    assert Transform("map", "robot_2/odom") in grown
    assert Transform("robot_2/base_link", "robot_2/laser") in grown


def test_build_tf_message_carries_the_links_as_transform_stamped():
    message = build_tf_message(
        (Transform("map", "odom"), Transform("base_link", "laser")),
        lambda name: codec.TYPESTORE.types[name],
    )

    assert [stamped.header.frame_id for stamped in message.transforms] == ["map", "base_link"]
    assert [stamped.child_frame_id for stamped in message.transforms] == ["odom", "laser"]
    for stamped in message.transforms:
        assert stamped.transform.translation.x == pytest.approx(0.0)
        assert stamped.transform.rotation.w == pytest.approx(1.0)


# -- the wiring -----------------------------------------------------------


def test_the_gateway_publishes_the_tree_the_session_implies(mirror):
    gateway, bridge, support = mirror
    bridge.emit(
        topics.ODOM, unity_odometry(x=1.5, y=0.5, yaw=0.25, frame_id="/odom", stamp=2.0)
    )
    bridge.emit(topics.SCAN, unity_laserscan([1.0, 2.0], frame_id="/laser", stamp=2.0))
    bridge.emit(topics.MAP, unity_occupancy_grid([[0, 0], [0, 0]], frame_id="/map", stamp=2.0))

    node = support.rclpy.nodes[0]
    dynamic = node.publishers["/tf"]
    assert dynamic.qos == 10, "the dynamic topic takes the default profile"
    assert len(dynamic.published) == 1
    stamped = dynamic.published[0].transforms
    assert [(entry.header.frame_id, entry.child_frame_id) for entry in stamped] == [
        ("odom", "base_link")
    ]
    assert stamped[0].transform.translation.x == pytest.approx(1.5)
    assert stamped[0].header.stamp.sec == 2

    static = node.publishers["/tf_static"]
    assert static.qos["depth"] == 1, "the static topic is latched, not a stream"
    assert static.qos["durability"] == "transient_local"
    assert [
        (entry.header.frame_id, entry.child_frame_id) for entry in static.published[-1].transforms
    ] == [("map", "odom"), ("base_link", "laser")]
    assert gateway.stats()["tf"] == 1
    assert gateway.stats()["tf_static"] == 2, "the lidar link, then the whole tree"
    assert gateway.last_error is None


def test_the_gateway_steps_aside_when_the_session_authors_tf(mirror):
    gateway, bridge, support = mirror
    bridge._published[topics.base("/tf")] = TF_MESSAGE_TYPE
    bridge.emit(topics.ODOM, unity_odometry(frame_id="/odom"))
    bridge.emit(topics.SCAN, unity_laserscan([1.0], frame_id="/laser"))
    bridge.emit(topics.MAP, unity_occupancy_grid([[0]], frame_id="/map"))

    node = support.rclpy.nodes[0]
    assert "/tf" not in node.publishers, "the app publishes /tf; the synthesis stays out of it"
    assert gateway.stats()["tf"] == 0
    assert "/tf_static" in node.publishers, "the app claimed only /tf; /tf_static is still ours"
    assert gateway.stats()["tf_static"] >= 1
    assert gateway.last_error is None
