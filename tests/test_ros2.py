"""The ROS2 transport of the viewer, exercised on a real graph.

The whole module is skipped when this interpreter has no ``rclpy``: a ROS2 install is a thing a
machine has or does not have, and a suite that failed without one would be reporting the machine
rather than the code. When one is present, the test publishes the four streams the viewer draws on a
real graph, in the same process, and reads them back through the same surface the bridge presents.
"""

import json
import math
import time

import pytest

from robotsnap import topics

rclpy = pytest.importorskip("rclpy", reason="no ROS2 install is sourced on this interpreter")
Odometry = pytest.importorskip("nav_msgs.msg").Odometry
OccupancyGrid = pytest.importorskip("nav_msgs.msg").OccupancyGrid
LaserScan = pytest.importorskip("sensor_msgs.msg").LaserScan
String = pytest.importorskip("std_msgs.msg").String


STATE = {
    "simulation_state": "running",
    "playing": True,
    "scenario_name": "Front Approach",
    "scenario_id": "front_approach",
    "sim_time_seconds": 12.5,
    "time_scale": 1.0,
    "map_name": "basic/frontal",
    "map_width": 2,
    "map_height": 2,
    "map_resolution": 0.5,
    "map_origin_x": -1.0,
    "map_origin_y": -1.0,
    "humans": [
        {"id": 1, "x": 3.0, "y": 0.0, "z": 0.0, "visible": True, "speed": 0.9},
    ],
    "robot_has_goal": True,
    "robot_goal": {"x": 9.0, "y": 0.0, "z": 0.0},
}

AGENTS = {"agents": [{"id": 1, "x": 2.0, "y": 0.0, "visible": True}]}


def _publish_until_read(publisher_node, client, topic, read, timeout=15.0):
    """Publish nothing ourselves: the caller published already, this waits for the match to happen."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        rclpy.spin_once(publisher_node, timeout_sec=0.02)
        value = read()
        if value is not None:
            return value
    return None


@pytest.fixture
def session():
    """A viewer client on the graph, with a publisher node beside it."""
    from robotsnap_ros2 import Ros2Client

    client = Ros2Client(node_name="robotsnap_test_viewer", fresh_seconds=60.0)
    publisher = rclpy.create_node("robotsnap_test_publisher")
    try:
        yield client, publisher
    finally:
        publisher.destroy_node()
        client.stop()


def test_the_session_reads_the_same_streams_the_bridge_would(session):
    client, publisher = session

    # The four streams the viewer draws, published the way Unity publishes them.
    odom_pub = publisher.create_publisher(Odometry, topics.ODOM, 10)
    scan_pub = publisher.create_publisher(LaserScan, topics.SCAN, 10)
    map_pub = publisher.create_publisher(OccupancyGrid, topics.MAP, 10)
    state_pub = publisher.create_publisher(String, topics.SIMULATION_STATE, 10)
    agents_pub = publisher.create_publisher(String, topics.SIMULATION_AGENTS, 10)

    # Subscriptions are matched asynchronously, so nothing is published until the graph has paired the
    # two nodes: this is the one thing a test on a real graph has to wait for.
    deadline = time.monotonic() + 30.0
    while time.monotonic() < deadline and odom_pub.get_subscription_count() == 0:
        rclpy.spin_once(publisher, timeout_sec=0.05)
        client.poll()

    odom = Odometry()
    odom.pose.pose.position.x = 4.0
    odom.pose.pose.position.y = -2.0
    odom.pose.pose.orientation.w = math.cos(0.25)
    odom.pose.pose.orientation.z = math.sin(0.25)
    odom.twist.twist.linear.x = 0.8
    odom_pub.publish(odom)

    scan = LaserScan()
    scan.angle_min = 0.0
    scan.angle_increment = 0.5
    scan.range_min = 0.1
    scan.range_max = 10.0
    scan.ranges = [1.0, 2.0, 3.0]
    scan_pub.publish(scan)

    grid = OccupancyGrid()
    grid.info.width = 2
    grid.info.height = 2
    grid.info.resolution = 0.5
    grid.info.origin.position.x = -1.0
    grid.info.origin.position.y = -1.0
    grid.data = [0, 0, 100, 0]
    map_pub.publish(grid)

    state_pub.publish(String(data=json.dumps(STATE)))
    agents_pub.publish(String(data=json.dumps(AGENTS)))

    # -- what the viewer reads -------------------------------------------

    read = _publish_until_read(publisher, client, topics.ODOM, lambda: client.last_message(topics.ODOM))
    assert read is not None, "the odometry subscription never matched"
    assert (read.pose.pose.position.x, read.pose.pose.position.y) == (4.0, -2.0)
    assert client.last_message(topics.ODOM) is not None, "the bare name addresses the primary robot"

    body = _publish_until_read(publisher, client, topics.SIMULATION_STATE, client.snapshot)
    assert body is not None and body["scenario_name"] == "Front Approach"
    assert body["sim_time_seconds"] == 12.5
    assert [human["id"] for human in body["humans"]] == [1]

    agents = _publish_until_read(publisher, client, topics.SIMULATION_AGENTS, client.agents)
    assert agents == AGENTS, "the agents body is handed over as the dict the viewer reads"

    state_message = _publish_until_read(
        publisher, client, topics.SCAN, lambda: client.bridge.snapshot().laser
    )
    assert state_message is not None, "the typed snapshot is built from the same messages"
    points = state_message.points_2d()
    assert len(points) == 3, "every range inside [range_min, range_max] is a point"
    assert points[0] == pytest.approx((1.0, 0.0))

    typed = client.bridge.snapshot()
    assert typed.robot is not None and typed.robot.x == pytest.approx(4.0)
    assert typed.simulation is not None and typed.simulation.scenario_id == "front_approach"
    assert typed.map_received is True
    assert client.bridge.topic_counts()[topics.base(topics.MAP)] >= 1
    assert client.bridge.port == 0, "a graph is not dialled"
    assert client.is_connected is True
    assert "ROS2" in client.session_label
    assert client.last_error is None


def test_the_graph_carries_no_port_and_says_so():
    """The viewer prints where it reads from; a ROS2 session has no port to name."""
    from robotsnap_ros2 import Ros2Client

    client = Ros2Client(node_name="robotsnap_test_label")
    try:
        assert client.bridge.port == 0
        assert client.session_label.startswith("the ROS2 graph")
        assert client.is_connected is False, "nothing published yet, so nothing is connected"
        assert client.last_message(topics.ODOM) is None
        assert client.snapshot() is None
    finally:
        client.stop()
