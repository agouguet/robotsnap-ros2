"""The ROS2 transport as a *writer*, exercised on a real graph.

``test_ros2_write.py`` drives this surface on a fake ``rclpy`` and checks the bytes; this file
checks the other half of the claim on a live graph, in one process: the :class:`Ros2Client` behind
``RobotSNAPClient(bridge=...)`` must put a ``geometry_msgs/Twist`` where the rest of the graph hears
it, put the JSON command on ``/simulation/control`` and find the acknowledgement on
``/simulation/control_result``. It also pins what "ready" means for a graph: ``wait_until_ready`` is
true only once a node of the graph has subscribed to ``/simulation/control``.

The whole module is skipped without a ROS2 install. The graph is shared with whatever else runs on
this machine - a ``controller`` node publishes null Twists on ``/cmd_vel`` - so nothing here asserts
on "the next message": arrivals are filtered by value and every wait is bounded.
"""

from __future__ import annotations

import json
import threading
import time

import pytest

from robotsnap import topics
from robotsnap.client import RobotSNAPClient
from robotsnap_ros2 import Ros2Client

rclpy = pytest.importorskip("rclpy", reason="no ROS2 install is sourced on this interpreter")
pytest.importorskip("nav_msgs.msg", reason="no ROS2 message packages on this interpreter")
Twist = pytest.importorskip(
    "geometry_msgs.msg", reason="no ROS2 message packages on this interpreter"
).Twist
String = pytest.importorskip(
    "std_msgs.msg", reason="no ROS2 message packages on this interpreter"
).String

#: The command the live tests drive; anything else on ``/cmd_vel`` belongs to the shared graph.
WANTED_TWIST = (0.5, -0.2)

#: What this test answers a ``play`` with, shaped exactly like the endpoint's acknowledgement.
PLAY_ACK = {"ok": True, "command": "play"}


class _GraphPeer:
    """One node of the test on the graph beside the client, with an executor of its own.

    The peer is spun from its own thread so the test can hear what the client wrote while the client
    is itself blocked waiting for an acknowledgement - both sides have to make progress at once. Its
    name is deliberately unlike the client's so nothing on the shared graph is mistaken for it.
    """

    def __init__(self, name: str) -> None:
        self.node = rclpy.create_node(name)
        self.executor = rclpy.executors.SingleThreadedExecutor()
        self.executor.add_node(self.node)
        self.thread = threading.Thread(target=self.executor.spin, daemon=True)
        self.thread.start()
        self._publishers: dict[str, object] = {}
        self._subscriptions: list[object] = []

    def publisher(self, message_type, topic: str):
        publisher = self._publishers.get(topic)
        if publisher is None:
            publisher = self.node.create_publisher(message_type, topic, 10)
            self._publishers[topic] = publisher
        return publisher

    def subscribe(self, message_type, topic: str, callback):
        subscription = self.node.create_subscription(message_type, topic, callback, 10)
        self._subscriptions.append(subscription)
        return subscription

    def close(self) -> None:
        self.executor.shutdown()
        self.thread.join(timeout=5.0)
        self.node.destroy_node()


@pytest.fixture
def session():
    """A ROS2 client and a graph peer of the test, both live in this interpreter."""
    client = Ros2Client(node_name="robotsnap_test_ros2_writer")
    peer = _GraphPeer("robotsnap_test_ros2_writer_peer")
    try:
        yield client, peer
    finally:
        peer.close()
        client.stop()


def _wait_for(predicate, timeout: float, what: str) -> None:
    """Wait for something the graph settles asynchronously, then fail naming what never happened."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.02)
    pytest.fail(what)


def test_a_cmd_vel_sent_through_the_facade_reaches_a_twist_subscriber(session):
    client, peer = session
    heard: list[tuple[float, float]] = []

    def record(message) -> None:
        # Filtered by value on purpose: the shared graph has its own /cmd_vel publisher, so only the
        # command this test sent proves the client wrote to the graph.
        values = (message.linear.x, message.angular.z)
        if values == WANTED_TWIST:
            heard.append(values)

    peer.subscribe(Twist, topics.CMD_VEL, record)

    facade = RobotSNAPClient(bridge=client.bridge)

    deadline = time.monotonic() + 30.0
    while time.monotonic() < deadline and not heard:
        # A publisher made on first use is paired with the subscriber asynchronously, so the command
        # is repeated until the graph carries it rather than assumed to survive the DDS match.
        if not facade.send_cmd_vel(*WANTED_TWIST):
            pytest.fail(facade.last_error or "send_cmd_vel returned False")
        time.sleep(0.02)

    assert heard, f"no Twist {WANTED_TWIST} reached the subscriber; saw {heard}"
    assert heard[-1] == WANTED_TWIST, f"the twist arrived altered: {heard[-1]}"


def test_play_publishes_the_command_and_reads_the_acknowledgement(session):
    client, peer = session
    commands: list[dict] = []

    ack_publisher = peer.publisher(String, topics.SIMULATION_CONTROL_RESULT)

    def answer(message) -> None:
        commands.append(json.loads(message.data))
        ack_publisher.publish(String(data=json.dumps(PLAY_ACK)))

    peer.subscribe(String, topics.SIMULATION_CONTROL, answer)

    # The client subscribes to the result topic from its constructor, so the pair has to be matched
    # before play() asks for its answer: one sent earlier would be heard by nobody.
    _wait_for(
        lambda: ack_publisher.get_subscription_count() > 0,
        15.0,
        "the client never subscribed to /simulation/control_result",
    )

    facade = RobotSNAPClient(bridge=client.bridge)
    # The first write can be dropped while the graph pairs the two nodes; the retry is bounded.
    facade.command_timeout = 1.0
    deadline = time.monotonic() + 30.0
    result = None
    while result is None and time.monotonic() < deadline:
        result = facade.play()

    assert commands and commands[-1] == {"command": "play"}, f"the graph carried {commands!r}"
    assert result == PLAY_ACK, facade.last_error


def test_wait_until_ready_is_the_graph_seeing_a_control_subscriber(session):
    client, peer = session

    if client.count_subscribers(topics.SIMULATION_CONTROL) > 0:
        pytest.skip("the shared graph already has a /simulation/control subscriber")

    # A current state stream is the other half of "connected": it is kept fresh for the whole test so
    # the control subscription stays the only thing that can decide readiness.
    state = json.dumps({"playing": True, "simulation_state": "running"})
    state_publisher = peer.publisher(String, topics.SIMULATION_STATE)
    peer.node.create_timer(0.1, lambda: state_publisher.publish(String(data=state)))
    _wait_for(lambda: client.is_connected, 15.0, "the client never heard the state stream")

    facade = RobotSNAPClient(bridge=client.bridge)

    # Nothing on the graph listens for commands yet, so a fresh state stream alone is not ready.
    assert facade.wait_until_ready(timeout=1.0) is False
    assert topics.SIMULATION_CONTROL in (facade.last_error or ""), facade.last_error

    peer.subscribe(String, topics.SIMULATION_CONTROL, lambda message: None)

    assert facade.wait_until_ready(timeout=15.0) is True, facade.last_error
