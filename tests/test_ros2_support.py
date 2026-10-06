"""How a message type name becomes a class, and why the package part is kept.

``Ros2Support`` is the one place a name is turned into something the graph can publish. The
interesting case is a name whose last segment matches a type this package already holds: answering it
from the short spelling would silently publish the wrong message, which is worse than refusing. The
tests below need no ROS2 install; the import path is asserted separately in ``test_ros2.py``.
"""

from __future__ import annotations

import pytest

from robotsnap_ros2 import Ros2Support, Ros2Unavailable


def test_a_full_name_is_not_answered_by_a_matching_short_one():
    support = Ros2Support(None, {"Odometry": "nav-odometry-class"})

    # ``other_msgs`` is not installed here, so this has to fail rather than hand back the nav_msgs
    # class because the last segment happens to match.
    with pytest.raises(Ros2Unavailable):
        support.get_message("other_msgs/msg/Odometry")


def test_a_short_name_the_package_knows_is_answered_from_the_table():
    odometry = object()
    support = Ros2Support(None, {"Odometry": odometry})

    assert support.get_message("Odometry") is odometry
    assert support.get_message("Odometry") is odometry, "resolved names are cached"


def test_a_short_name_the_package_does_not_know_is_refused_with_the_way_out():
    support = Ros2Support(None, {})

    with pytest.raises(Ros2Unavailable) as raised:
        support.get_message("Nonsense")
    assert "pkg/msg/Type" in str(raised.value)


def test_a_name_that_is_not_a_message_type_is_refused():
    support = Ros2Support(None, {})

    with pytest.raises(Ros2Unavailable):
        support.get_message("nav_msgs/msg/Odometry/extra")
    with pytest.raises(Ros2Unavailable):
        support.get_message("nav_msgs/Odometry")
