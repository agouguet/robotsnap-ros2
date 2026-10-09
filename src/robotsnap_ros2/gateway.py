"""The Unity endpoint's other side: the same session, mirrored on a ROS2 graph.

``python -m robotsnap.bridge --ros2`` runs this beside the TCP endpoint, so Unity dials this package
instead of ``ros_tcp_endpoint``. Unity keeps speaking the connector protocol on its socket; whatever
the session publishes is put on the graph for the navigation methods to read, and whatever the graph
sends to the session is written back on the socket. Nothing above the bridge changes: the client, the
Gymnasium environment and the viewer read the same bridge they always read, and never learn that a
graph is attached. That is the separation this module exists for - the transport is a choice the
bridge makes, not a fact the application has to know.

The mirror copies no field. A payload Unity sent is the CDR that
:func:`robotsnap.bridge.codec.encode` wrote, ``rclpy.serialization.deserialize_message`` reads it
back with the same values, and the graph republishes what that decode produced. ``rosbags`` pads a
message's alignment gaps with zeros where ``rclpy`` leaves them uninitialised, so the two encodings
are equal in every field and may differ in a padding byte; a mirror that copied fields would have
more ways to be wrong, not fewer.

Which topics cross is not a list written here: it is what Unity said about itself. Every stream the
peer *publishes* is mirrored onto the graph, and every stream the peer *subscribed to* is subscribed
on the graph and forwarded to it, so a build that adds a topic gets it bridged without this module
changing - while a topic Unity only listens to is never offered on the graph as something it sends.

One thing is added rather than mirrored: the transforms the session's own streams imply. Unity names
the frames its ``/odom``, ``/scan`` and ``/map`` sit in but publishes no ``/tf``, so a graph reader has
three streams and no idea how they relate - RViz2 draws nothing and a navigation stack cannot place
the robot. :mod:`robotsnap_ros2.tf` derives the links from those messages and this module publishes
them on ``/tf`` and ``/tf_static``, latched where it has to be so a late subscriber still gets them.
The synthesis only fills a name the session left empty: when the peer publishes a ``/tf`` or
``/tf_static`` of its own, the application is the author of that tree and this module steps aside.
"""

from __future__ import annotations

import threading
import time
from typing import Any

from robotsnap import topics
from robotsnap_ros2.support import Ros2Unavailable, Ros2Support, ros2_support
from robotsnap_ros2.tf import TF_MESSAGE_TYPE, TfComposer, build_tf_message, is_odometry

__all__ = ["DISCOVERY_SECONDS", "Ros2Gateway"]

#: How often the mirror re-reads what Unity announced, and what the graph now offers. Cheap enough to
#: run while nothing changes, slow enough not to make the graph query the hot path.
DISCOVERY_SECONDS = 0.5

#: The two transform topics the gateway may synthesise. A session that publishes either of them keeps
#: it: the app's tree wins, and the gateway only fills the name the app left empty.
TF_TOPIC = "/tf"
TF_STATIC_TOPIC = "/tf_static"

#: The two directions a topic can be refused in. They are separate questions about one name, so the
#: refusals are remembered separately and reported with this word in front.
_PUBLISHING = "publishing"
_SUBSCRIBING = "subscribing"


class Ros2Gateway:
    """Publishes what Unity sends, and sends back what the graph publishes.

    It owns a node and the thread that spins it, and it listens to the bridge for the session's own
    messages. Both directions are best-effort by design: a graph that is not there, a type the graph
    does not know and a listener that raised all leave the session running, which is what a component
    sitting *beside* a simulation owes it.
    """

    def __init__(self, bridge, *, node_name: str = "robotsnap_gateway"):
        support: Ros2Support = ros2_support()
        self._support = support
        self._bridge = bridge
        self._started_context = not support.rclpy.ok()
        if self._started_context:
            support.rclpy.init(args=None)

        self._node = support.rclpy.create_node(node_name)
        # A spin of its own: the viewer or a test node may be spinning in this same interpreter, and
        # two threads on the process-wide executor raise instead of sharing it.
        self._executor = support.new_executor()
        self._lock = threading.RLock()
        self._publishers: dict[str, Any] = {}
        self._tf_publishers: dict[str, Any] = {}
        self._subscriptions: dict[str, Any] = {}
        self._skipped: dict[str, dict[str, str]] = {_PUBLISHING: {}, _SUBSCRIBING: {}}
        self._error: str | None = None
        self._closed = False
        self._discovered_at = 0.0
        self._spin_thread: threading.Thread | None = None
        self._composer = TfComposer()
        self._counts = {"published": 0, "received": 0, "skipped": 0, "tf": 0, "tf_static": 0}

        bridge.add_message_listener(self._on_session_message)

    # -- lifecycle ----------------------------------------------------------

    def start(self) -> None:
        """Register what the session already announced, then spin the node on its own thread."""
        self._discover()
        if self._executor is not None:
            self._executor.add_node(self._node)
        thread = threading.Thread(
            target=self._spin_loop, name="robotsnap-ros2-gateway", daemon=True
        )
        self._spin_thread = thread
        thread.start()

    def stop(self) -> None:
        """Leave the graph as this gateway found it. Idempotent."""
        with self._lock:
            if self._closed:
                return
            self._closed = True
            thread = self._spin_thread
            self._spin_thread = None
        self._bridge.remove_message_listener(self._on_session_message)
        if thread is not None and thread.is_alive() and thread is not threading.current_thread():
            thread.join(timeout=2.0)
        if self._executor is not None:
            try:
                self._executor.remove_node(self._node)
                self._executor.shutdown()
            except Exception as exc:  # a context already gone is not this caller's problem
                self._error = self._error or str(exc)
        try:
            self._node.destroy_node()
        except Exception as exc:  # a context already gone is not this caller's problem
            self._error = self._error or str(exc)
        finally:
            if self._started_context and self._support.rclpy.ok():
                self._support.rclpy.shutdown()

    @property
    def closed(self) -> bool:
        return self._closed

    @property
    def last_error(self) -> str | None:
        return self._error

    def stats(self) -> dict[str, int]:
        """How many messages crossed each way, for the bridge's status line."""
        with self._lock:
            return dict(self._counts)

    def skipped(self) -> dict[str, str]:
        """Topics the graph could not take, and why - a type it does not have, mainly.

        Keyed by ``"<topic> (<direction>)"``, because one topic can be refused one way while it still
        crosses the other, and a caller reading the recap has to see which one stopped.
        """
        with self._lock:
            return {
                f"{key} ({direction})": reason
                for direction, refused in self._skipped.items()
                for key, reason in refused.items()
            }

    # -- Unity -> graph -----------------------------------------------------

    def _on_session_message(self, key: str, payload: bytes, msg_type: str) -> None:
        """Put one message the session published onto the graph.

        Called from the bridge's read thread, so it does the smallest thing that can work: look up the
        publisher, turn the payload back into a message, publish. Creating a publisher can block the
        first time - it waits for the graph - which is why it is done once per topic and not per
        message.
        """
        if self._closed:
            return
        made = self._publisher_for(key, msg_type)
        if made is None:
            return
        publisher, message_type = made
        try:
            # Unity's payload is CDR and rclpy reads it directly. The one field this does touch is the
            # frame a message names, because the session names frames with a leading slash and tf2
            # rejects that outright; see _strip_frame_prefix.
            message = self._support.deserialize(payload, message_type)
            _strip_frame_prefix(message)
            publisher.publish(message)
        except Exception as exc:
            self._record(f"could not mirror {key}: {exc}")
            return
        with self._lock:
            self._counts["published"] += 1
        self._publish_transforms(key, msg_type, message)

    def _publisher_for(self, key: str, msg_type: str):
        """The publisher for one session topic, made once, with the type it carries."""
        with self._lock:
            publisher = self._publishers.get(key)
            if publisher is not None:
                return publisher
            if key in self._skipped[_PUBLISHING]:
                return None

        if not msg_type:
            self._skip(_PUBLISHING, key, "the session did not say which type it publishes")
            return None
        try:
            message_type = self._support.get_message(msg_type)
        except Exception as exc:
            self._skip(_PUBLISHING, key, f"the graph has no message type {msg_type}: {exc}")
            return None

        topic_name = "/" + key
        # ``/map`` is a state, not a stream: a navigation stack subscribes to it with a
        # transient-local durability so that a late-joining costmap still gets the map. A
        # volatile publisher never matches that subscription, and nav2 reports the silence
        # as "Can't update static costmap layer, no map received" - then falls back to a
        # default five-metre costmap with the robot outside it. Latched here for the same
        # reason ``/tf_static`` is.
        qos = self._transform_qos(latched=True) if key == topics.base(topics.MAP) else 10
        try:
            publisher = self._node.create_publisher(message_type, topic_name, qos)
        except Exception as exc:
            self._skip(_PUBLISHING, key, f"could not publish on {topic_name}: {exc}")
            return None
        with self._lock:
            self._publishers[key] = (publisher, message_type)
        return publisher, message_type

    # -- Unity -> graph: the transforms the streams imply -------------------

    def _publish_transforms(self, key: str, msg_type: str, message) -> None:
        """Publish what one session message implies on ``/tf`` and ``/tf_static``.

        Odometry yields the dynamic link, and a laser scan or an occupancy grid yields the frames the
        static links need; any other stream leaves the tree as it was. The static side is asked for on
        every one of them because the frames arrive in an order nobody controls - a scan before the
        odometry it hangs off is normal - and :meth:`TfComposer.new_static_transforms` answers nothing
        when the tree has not changed. Kept best-effort like the mirror itself: a graph that refuses a
        transform is recorded here and never stops the session.
        """
        try:
            self._composer.observe(key, msg_type, message)
            if is_odometry(msg_type):
                self._publish_dynamic()
            self._publish_static()
        except Exception as exc:
            self._record(f"could not publish the transforms of {key}: {exc}")

    def _publish_dynamic(self) -> None:
        """Put the dynamic links on ``/tf``, unless the app authors ``/tf`` itself."""
        if self._app_authors(TF_TOPIC):
            return
        links = self._composer.dynamic_transforms()
        message = self._transform_message(links)
        if message is None:
            return
        self._publish_on(TF_TOPIC, message, latched=False, counter="tf")

    def _publish_static(self) -> None:
        """Put the new static links on ``/tf_static``, unless the app authors it itself."""
        if self._app_authors(TF_STATIC_TOPIC):
            return
        links = self._composer.new_static_transforms()
        message = self._transform_message(links)
        if message is None:
            return
        self._publish_on(TF_STATIC_TOPIC, message, latched=True, counter="tf_static")

    def _app_authors(self, topic: str) -> bool:
        """Whether the session publishes this transform topic itself, in which case the app wins.

        The peer may already carry a ``/tf`` - a Unity build that broadcasts its own tree, or a
        scenario that fixes a calibration - and the mirror republishes that like any other stream. A
        second publisher here would put two different trees on one name, so the synthesis steps aside
        for the topic the app claimed. The question is asked of what the peer *publishes*: a session
        that only *listens* to ``/tf`` is a consumer of this synthesis, not a rival author of it.
        """
        return topics.base(topic) in self._bridge.published_topics()

    def _transform_message(self, links):
        """The ``TFMessage`` carrying ``links``, or None when there is nothing to carry."""
        if not links:
            return None
        try:
            return build_tf_message(links, self._support.get_message)
        except Exception as exc:
            self._record(f"could not build a transform message: {exc}")
            return None

    def _publish_on(self, topic: str, message, *, latched: bool, counter: str) -> None:
        """Publish one transform message, creating its publisher on the first call."""
        publisher = self._transform_publisher(topic, latched=latched)
        if publisher is None:
            return
        try:
            publisher.publish(message)
        except Exception as exc:
            self._record(f"could not publish on {topic}: {exc}")
            return
        with self._lock:
            self._counts[counter] += 1

    def _transform_publisher(self, topic: str, *, latched: bool):
        """The publisher of ``/tf`` or ``/tf_static``, made once.

        ``/tf`` takes the default profile and a depth of ten, like every mirrored stream. ``/tf_static``
        is published *latched* - transient-local, reliable, depth one - so a reader that joins after
        the links were published, RViz2 opened late or a navigation stack started after the map, still
        receives them; without that, a static link is heard only by whoever was already listening when
        it crossed, which for a tree published once is nobody.
        """
        with self._lock:
            publisher = self._tf_publishers.get(topic)
        if publisher is not None:
            return publisher
        try:
            message_type = self._support.get_message(TF_MESSAGE_TYPE)
        except Exception as exc:
            self._record(f"the graph has no {TF_MESSAGE_TYPE}: {exc}")
            return None
        try:
            publisher = self._node.create_publisher(
                message_type, topic, self._transform_qos(latched=latched)
            )
        except Exception as exc:
            self._record(f"could not publish on {topic}: {exc}")
            return None
        with self._lock:
            self._tf_publishers[topic] = publisher
        return publisher

    def _transform_qos(self, *, latched: bool):
        """The QoS of one transform topic: depth ten, or the latched profile for ``/tf_static``.

        A support whose ``rclpy`` exposes no QoS module still gets depth one for the static topic,
        because a ``transient_local`` profile is not something the rest of the transport can build;
        the graph then keeps the last message instead of latching it, the closest it can say.
        """
        if not latched:
            return 10
        qos = _rclpy_qos(self._support)
        if qos is None:
            return 1
        return qos.QoSProfile(
            depth=1,
            reliability=qos.ReliabilityPolicy.RELIABLE,
            durability=qos.DurabilityPolicy.TRANSIENT_LOCAL,
        )

    # -- graph -> Unity -----------------------------------------------------

    def _discover(self) -> None:
        """Open the streams that appeared since the last look, in both directions.

        The session's own registrations are the source of truth: a topic Unity *publishes* gets a
        publisher here, a topic it *subscribed to* gets a subscription. Reading the two apart matters,
        because a topic Unity only listens to must not be offered on the graph as a stream it sends -
        ``/cmd_vel`` above all. Querying the graph's own topic list is only needed for a fleet's
        namespaced commands, whose names exist in the graph before the session has said anything about
        them.
        """
        for key, msg_type in self._bridge.published_topics().items():
            if key not in self._publishers and key not in self._skipped[_PUBLISHING]:
                self._publisher_for(key, msg_type)

        for key, name in self._bridge.subscriptions().items():
            self._listen(name, self._bridge.topic_types().get(key))

        try:
            offered = self._node.get_topic_names_and_types()
        except Exception as exc:
            self._record(f"could not list the graph: {exc}")
            return
        for name, types in offered:
            key = topics.base(name)
            if key in self._bridge.subscriptions() or not types:
                continue
            # A namespaced command another node publishes before the session has announced its own
            # subscription: the type is the graph's, and the name is what Unity will be sent it on.
            if key.endswith("/" + topics.base(topics.CMD_VEL)) or key.endswith(
                "/" + topics.base(topics.SIMULATION_CONTROL)
            ):
                self._listen(name, types[0])

    def _listen(self, topic: str, msg_type: str | None) -> None:
        """Subscribe on the graph to one stream, and forward what it carries to the session."""
        key = topics.base(topic)
        with self._lock:
            if key in self._subscriptions or key in self._skipped[_SUBSCRIBING]:
                return
        if not msg_type:
            self._skip(_SUBSCRIBING, key, "the session did not say which type it listens to")
            return
        try:
            message_type = self._support.get_message(msg_type)
        except Exception as exc:
            self._skip(_SUBSCRIBING, key, f"the graph has no message type {msg_type}: {exc}")
            return

        def forward(message, *, _topic=topic) -> None:
            self._forward(_topic, message)

        try:
            subscription = self._node.create_subscription(message_type, topic, forward, 10)
        except Exception as exc:
            self._skip(_SUBSCRIBING, key, f"could not subscribe to {topic}: {exc}")
            return
        with self._lock:
            self._subscriptions[key] = subscription

    def _forward(self, topic: str, message) -> None:
        """Write one graph message back on the socket, under the name Unity registered."""
        if self._closed:
            return
        try:
            payload = self._support.serialize(message)
        except Exception as exc:
            self._record(f"could not serialize a message on {topic}: {exc}")
            return
        if not self._bridge.send_message(self._bridge.destination_for(topic), payload):
            # Nothing is connected: the command is dropped rather than queued, because a velocity
            # command that reaches the session a second later is worse than one that never arrived.
            return
        with self._lock:
            self._counts["received"] += 1

    # -- spin ---------------------------------------------------------------

    def _spin_loop(self) -> None:
        while not self._closed and self._support.rclpy.ok():
            try:
                if self._executor is not None:
                    self._executor.spin_once(timeout_sec=0.05)
                else:
                    self._support.rclpy.spin_once(self._node, timeout_sec=0.05)
            except Exception as exc:  # a context shut down under us ends the loop, quietly
                self._record(f"the graph stopped: {exc}")
                return
            now = time.monotonic()
            if now - self._discovered_at >= DISCOVERY_SECONDS:
                self._discovered_at = now
                self._discover()

    # -- bookkeeping --------------------------------------------------------

    def _skip(self, direction: str, key: str, reason: str) -> None:
        """Remember a stream the graph cannot carry, so the reason is said once instead of per message.

        Direction is part of the key. Publishing and subscribing are two separate questions about one
        topic name - a type the graph cannot publish says nothing about whether it can subscribe - and
        a shared memo would let the first refusal silence the other side.
        """
        with self._lock:
            if key in self._skipped[direction]:
                return
            self._skipped[direction][key] = reason
            self._counts["skipped"] += 1

    def _record(self, message: str) -> None:
        self._error = str(message)


def _strip_frame_prefix(message) -> None:
    """Drop the leading slash from the frames a mirrored message names.

    The session names its frames ``/odom``, ``/map`` and ``/laser``, and tf2
    refuses any frame id that starts with a slash: the check lives in ``libtf2``
    and a reader that is handed one reports "frame_ids cannot start with a '/'"
    rather than looking anything up. A navigation stack then fails in ways that
    look like something else entirely - a costmap that never receives its map
    ("Can't update static costmap layer, no map received") and a laser that never
    reaches its observation buffer, both because the frame those messages name
    does not resolve.

    The transform tree this gateway publishes is already normalised by
    :func:`robotsnap_ros2.tf.normalise_frame`; this does the same for the messages
    themselves, so a message and the tree agree on what a frame is called. Only a
    leading slash is touched: a name without one is passed through untouched, and
    a message that names no frame is left alone.
    """
    header = getattr(message, "header", None)
    frame = getattr(header, "frame_id", None)
    if isinstance(frame, str) and frame.startswith("/"):
        header.frame_id = frame.lstrip("/")
    child = getattr(message, "child_frame_id", None)
    if isinstance(child, str) and child.startswith("/"):
        message.child_frame_id = child.lstrip("/")


def unavailable_reason() -> str | None:
    """Why no gateway can run here, or None when one can."""
    try:
        ros2_support()
    except Ros2Unavailable as exc:
        return str(exc)
    return None


def _rclpy_qos(support: Ros2Support):
    """The ``rclpy.qos`` module, which ``import rclpy`` alone does not put in place.

    ``rclpy/__init__.py`` imports the node, the executor and the logging, not the QoS module: asking
    the module object for a ``qos`` attribute answers ``None`` on a real install while a stand-in that
    provides one answers it. Reading it that way round - the support's own attribute first, the real
    submodule second - is what lets a test hand in its own profile and a real run still latch
    ``/tf_static``. Guessing wrong costs the static tree its latch, silently, for every reader that
    joins after the map did.
    """
    qos = getattr(getattr(support, "rclpy", None), "qos", None)
    if qos is not None:
        return qos
    try:
        from rclpy import qos as module
    except ImportError:  # no ROS2 install: the caller falls back to a plain depth
        return None
    return module
