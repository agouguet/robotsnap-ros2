"""The transform tree a session implies, derived from the messages it publishes.

A robot says where it is (``/odom``), what it sees (``/scan``) and the world it moves in (``/map``),
but it names the *frames* those messages sit in and never the links between them. RViz2 and a
navigation stack need the links themselves: with nothing on ``/tf``, the map, the robot and the lidar
are three streams of numbers with no relation to each other, and the display can only say that a
frame has no transform.

This module derives the links a session implies - ``map -> odom``, ``odom -> base_link`` and
``base_link -> laser`` - from the messages alone, so a caller can publish them. It imports no
``rclpy`` and holds no ROS2 object: the composition is plain data, and only :func:`build_tf_message`
touches a message class, one the caller resolves. That is what lets the arithmetic be read and tested
on a machine with no ROS2 install, next to the rest of the unit suite.

Nothing here invents a value. A robot whose odometry carries no pose contributes no dynamic link, and
a session that has published no scan contributes no lidar link: the tree is only ever made of frames
and poses the session itself named.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable

from robotsnap import topics

__all__ = [
    "TF_MESSAGE_TYPE",
    "Transform",
    "TfComposer",
    "build_tf_message",
    "is_odometry",
    "normalise_frame",
]

#: The message type of a transform tree, the one both ``/tf`` and ``/tf_static`` carry.
TF_MESSAGE_TYPE = "tf2_msgs/msg/TFMessage"

#: Every static link of a session is the identity: the session names the *frames* a message sits in,
#: never a pose between two of them. Both constants are the canonical form of "no transform".
_ZERO_TRANSLATION = (0.0, 0.0, 0.0)
_IDENTITY_ROTATION = (0.0, 0.0, 0.0, 1.0)

#: The stamp of a static link. A static link's time carries no information - tf2 keeps it outside its
#: time-indexed buffer - and a stable value is what lets "has the tree changed" be a comparison of
#: content rather than a comparison a fresh timestamp would defeat on every message.
_STATIC_STAMP = (0, 0)

#: ``msg_type`` as the bridge normalises it, and as a caller may still spell it. Both are accepted so
#: the composer can be driven from a decoded message without the caller second-guessing the bridge.
_ODOMETRY_TYPES = frozenset({topics.ODOMETRY_TYPE, "nav_msgs/Odometry"})
_LASER_TYPES = frozenset({topics.LASER_SCAN_TYPE, "sensor_msgs/LaserScan"})
_MAP_TYPES = frozenset({topics.OCCUPANCY_GRID_TYPE, "nav_msgs/OccupancyGrid"})


def normalise_frame(name: Any) -> str:
    """The tf2 spelling of a frame name: no leading slash, no surrounding space, possibly empty.

    tf2 *refuses* a frame id that starts with ``/`` - the check lives in ``libtf2`` and a tree that
    carried one would be rejected wholesale - so every frame a message names goes through here before
    it is published. An unnamed frame answers ``""``: a frame the session did not give is a link that
    cannot be built, never a link with an invented name.
    """
    return str(name or "").strip().lstrip("/").strip()


@dataclass(frozen=True)
class Transform:
    """One link of the tree, as plain data, ready to become a ``TransformStamped``.

    The fields are tuples rather than message objects so two links compare by value: that comparison
    is what tells a caller whether the static part of the tree still has to be published. ``stamp`` is
    ``(sec, nanosec)``, and both geometry defaults are the identity - the shape of every static link.
    """

    parent: str
    child: str
    translation: tuple[float, float, float] = _ZERO_TRANSLATION
    rotation: tuple[float, float, float, float] = _IDENTITY_ROTATION
    stamp: tuple[int, int] = _STATIC_STAMP


@dataclass
class _Robot:
    """What one robot namespace has said about itself so far.

    Frames are kept apart from the dynamic link on purpose: a pose-less odometry still names the
    frames of the static links, and those stay buildable while no dynamic link exists yet.
    """

    odom_parent: str = ""
    body: str = ""
    sensor: str = ""
    odom: Transform | None = None


def _kind(msg_type: str) -> str | None:
    """Which of the three streams a message type belongs to, or None for any other stream."""
    name = str(msg_type).strip()
    if name in _ODOMETRY_TYPES:
        return "odometry"
    if name in _LASER_TYPES:
        return "scan"
    if name in _MAP_TYPES:
        return "map"
    return None


def is_odometry(msg_type: str) -> bool:
    """Whether ``msg_type`` names a ``nav_msgs/Odometry``, in either spelling the bridge may hand over.

    The bridge normalises Unity's ``nav_msgs/Odometry`` to ``nav_msgs/msg/Odometry``; a caller driving
    the composer directly may still spell it the first way. Both are one stream, and it is the only
    one that carries a dynamic link.
    """
    return str(msg_type).strip() in _ODOMETRY_TYPES


def _namespace(key: str) -> str:
    """The robot namespace of a topic key: ``robot_2/odom`` -> ``robot_2``, ``odom`` -> ``""``."""
    parts = topics.base(key).split("/")
    return "/".join(parts[:-1])


def _stamp_of(header: Any) -> tuple[int, int]:
    """``(sec, nanosec)`` of a message header, or the static stamp when it carries no time."""
    stamp = getattr(header, "stamp", None)
    if stamp is None:
        return _STATIC_STAMP
    try:
        return (int(getattr(stamp, "sec", 0)), int(getattr(stamp, "nanosec", 0)))
    except (TypeError, ValueError):
        return _STATIC_STAMP


def _translation_of(pose: Any) -> tuple[float, float, float]:
    position = getattr(pose, "position", None)
    return (
        float(getattr(position, "x", 0.0)),
        float(getattr(position, "y", 0.0)),
        float(getattr(position, "z", 0.0)),
    )


def _rotation_of(pose: Any) -> tuple[float, float, float, float]:
    orientation = getattr(pose, "orientation", None)
    return (
        float(getattr(orientation, "x", 0.0)),
        float(getattr(orientation, "y", 0.0)),
        float(getattr(orientation, "z", 0.0)),
        float(getattr(orientation, "w", 1.0)),
    )


class TfComposer:
    """Derives the links a session implies, robot by robot, from the messages it publishes.

    One instance follows one session. :meth:`observe` is handed each decoded message under the topic
    key it arrived on; the composer keeps, per robot namespace, the frames that message named, and
    :meth:`dynamic_transforms` and :meth:`new_static_transforms` answer the links they imply. Nothing
    is published here - the caller turns the links into a message and puts it on the graph.

    The map frame is kept per namespace with the primary one as a fallback: a fleet's odometry and
    scan are namespaced, the occupancy grid usually is not, and a robot that never published a map of
    its own still has to hang off the session's.
    """

    def __init__(self) -> None:
        self._robots: dict[str, _Robot] = {}
        self._maps: dict[str, str] = {}
        self._published_static: tuple[Transform, ...] = ()

    # -- observation --------------------------------------------------------

    def observe(self, key: str, msg_type: str, message: Any) -> None:
        """Take in one decoded session message and remember the frames it names.

        The three streams of the contract each contribute one thing. Odometry names both frames of
        the dynamic link and, when it carries a pose, the link itself. A laser scan names the sensor
        frame. An occupancy grid names the map frame. Any other topic is ignored, so a caller can
        hand every message over without deciding first.
        """
        kind = _kind(msg_type)
        if kind is None or message is None:
            return
        namespace = _namespace(key)
        if kind == "map":
            frame = normalise_frame(getattr(getattr(message, "header", None), "frame_id", ""))
            if frame:
                self._maps[namespace] = frame
            return

        robot = self._robots.setdefault(namespace, _Robot())
        header = getattr(message, "header", None)
        if kind == "scan":
            frame = normalise_frame(getattr(header, "frame_id", ""))
            if frame:
                robot.sensor = frame
            return

        parent = normalise_frame(getattr(header, "frame_id", ""))
        child = normalise_frame(getattr(message, "child_frame_id", ""))
        if parent:
            robot.odom_parent = parent
        if child:
            robot.body = child
        pose = getattr(getattr(message, "pose", None), "pose", None)
        if pose is None or not parent or not child:
            # No pose, or no named frame: there is no dynamic link to build, and none is invented. A
            # frame the session did name is kept, so the static links still become buildable.
            return
        robot.odom = Transform(
            parent=parent,
            child=child,
            translation=_translation_of(pose),
            rotation=_rotation_of(pose),
            stamp=_stamp_of(header),
        )

    # -- what the session implies ------------------------------------------

    def dynamic_transforms(self) -> tuple[Transform, ...]:
        """The dynamic links of the tree - today, the ``<odom> -> base_link`` of each robot.

        A robot whose odometry carried a pose has one; a robot that has published no odometry yet, or
        whose odometry named no frame, contributes nothing. A message that carried no pose leaves the
        last link in place rather than blanking the tree, so a consumer never loses the robot over one
        incomplete frame.
        """
        return tuple(
            robot.odom for _, robot in sorted(self._robots.items()) if robot.odom is not None
        )

    def static_transforms(self) -> tuple[Transform, ...]:
        """The static links of the tree, identity by construction.

        Two per robot: ``map -> <odom parent>`` and ``<body> -> <sensor>``. A link appears only once
        the two frames it joins are known, so the tree grows from the primary robot's namespace to a
        fleet's own without one robot's map being mistaken for another's.
        """
        links: list[Transform] = []
        for namespace, robot in sorted(self._robots.items()):
            map_frame = self._maps.get(namespace) or self._maps.get("")
            if map_frame and robot.odom_parent:
                links.append(Transform(map_frame, robot.odom_parent))
            if robot.body and robot.sensor:
                links.append(Transform(robot.body, robot.sensor))
        return tuple(links)

    def new_static_transforms(self) -> tuple[Transform, ...]:
        """The static links to publish now, or an empty tuple when they are the ones last handed out.

        This answers "is there anything new for ``/tf_static``": handing the same links back once per
        odometry message would republish the static half of the tree at the odometry rate. The
        comparison is over the whole link - frames, geometry and stamp - and the static stamp is
        fixed, so what decides is exactly the content ``/tf_static`` carries.
        """
        current = self.static_transforms()
        if current == self._published_static:
            return ()
        self._published_static = current
        return current


def build_tf_message(transforms: Iterable[Transform], get_message: Callable[[str], Any]):
    """A ``tf2_msgs/msg/TFMessage`` carrying ``transforms``.

    ``get_message`` is the resolver the rest of the transport already uses
    (``Ros2Support.get_message``): it answers a ``pkg/msg/Type`` name with a message class. Resolving
    from the outside is what keeps this module - and the composition above it - free of ``rclpy``,
    while both the graph's ``rclpy`` classes and the ``rosbags`` typestore accept the keyword fields
    built here.
    """
    message_class = get_message(TF_MESSAGE_TYPE)
    stamped_class = get_message("geometry_msgs/msg/TransformStamped")
    transform_class = get_message("geometry_msgs/msg/Transform")
    vector_class = get_message("geometry_msgs/msg/Vector3")
    quaternion_class = get_message("geometry_msgs/msg/Quaternion")
    header_class = get_message("std_msgs/msg/Header")
    time_class = get_message("builtin_interfaces/msg/Time")

    stamped = []
    for link in transforms:
        x, y, z = link.translation
        qx, qy, qz, qw = link.rotation
        sec, nanosec = link.stamp
        stamped.append(
            stamped_class(
                header=header_class(
                    stamp=time_class(sec=int(sec), nanosec=int(nanosec)),
                    frame_id=link.parent,
                ),
                child_frame_id=link.child,
                transform=transform_class(
                    translation=vector_class(x=float(x), y=float(y), z=float(z)),
                    rotation=quaternion_class(
                        x=float(qx), y=float(qy), z=float(qz), w=float(qw)
                    ),
                ),
            )
        )
    return message_class(transforms=stamped)
