"""The ROS2 transport of RobotSNAP, registered in the ``robotsnap.transports`` entry-point group.

The core imports no ROS2 code and knows this package only by the entry point it registers.
Installed beside the core, it is what the core loads when a run asks for ``ros2``:
:class:`Ros2Client` reads a session off a graph, :class:`Ros2Gateway` mirrors a socket session onto
one, and everything above them keeps speaking the surface it always did.
"""

from robotsnap_ros2.gateway import DISCOVERY_SECONDS, Ros2Gateway, unavailable_reason
from robotsnap_ros2.support import Ros2Client, Ros2Support, Ros2Unavailable, ros2_support

__all__ = [
    "DISCOVERY_SECONDS",
    "Ros2Client",
    "Ros2Gateway",
    "Ros2Support",
    "Ros2Unavailable",
    "ros2_support",
    "unavailable_reason",
]
