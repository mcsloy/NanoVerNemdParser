import time
import warnings
from os.path import join
import re
import glob
import numpy as np
from numpy.typing import NDArray
from concurrent import futures
from nanover.app import NanoverFrameApplication
from nanover.mdanalysis import mdanalysis_to_frame_data
from threading import RLock
import MDAnalysis
from MDAnalysis.coordinates.memory import MemoryReader
from nanover.trajectory.frame_server import PLAY_COMMAND_KEY, RESET_COMMAND_KEY, STEP_COMMAND_KEY, PAUSE_COMMAND_KEY


class TrajectoryPlayback:
    """MDAnalysis based trajectory plaback.

    A class to manage the playback of a molecular dynamics trajectory using MDAnalysis
    and NanoVer.

    This class provides methods to control the playback, such as playing, pausing,
    stepping through frames, and resetting the trajectory. It also handles the communication
    with the NanoVer frame server to update the particle positions in the VR environment.

    Attributes:
        universe: The MDAnalysis universe object containing the trajectory data.
        frame_server: The NanoVer frame server to send frame data to. If None, a new
            server is created.
        fps: The number of frames per second for the trajectory playback.
        colour_metric_normalisation_min: Minimum value for the normalisation
            of the color metric.
        colour_metric_normalisation_max: Maximum value for the normalisation
            of the color metric.
        colour_metric_normalisation_power: Power used for non-linear
            normalisation of the color metric.

    """

    def __init__(
            self, universe, frame_server=None, fps=60,
            colour_metric_normalisation_min: float = 0.0,
            colour_metric_normalisation_max: float = 1.0,
            colour_metric_normalisation_power: float = 1.0):
        """Initialises the TrajectoryPlayback object.

        Sets up the MDAnalysis universe, frame server, and frames per second (fps)
        for playback.

        Arguments:
            universe: The MDAnalysis universe object containing the trajectory data.
            frame_server: The NanoVer frame server to send frame data to. If None, a new
                server is created.
            fps: The number of frames per second for the trajectory playback.
            colour_metric_normalisation_min: Minimum value for the normalisation
                of the color metric.
            colour_metric_normalisation_max: Maximum value for the normalisation
                of the color metric.
            colour_metric_normalisation_power: Power used for non-linear
                normalisation of the color metric.
        """
        self.universe = universe

        if frame_server is None:
            self.frame_server = NanoverFrameApplication.basic_server(port=0)
        else:
            self.frame_server = frame_server

        self.set_up_commands()

        self.fps = fps

        # Get a pool of threads (just one) that we can run the playback on
        self.threads = futures.ThreadPoolExecutor(max_workers=1)
        self._run_task = None
        self._cancelled = False
        self._cancel_lock = RLock()
        self.frame_index = 0

        _ = universe.universe.trajectory[0]

        self.colour_metric_normalisation_min = colour_metric_normalisation_min
        self.colour_metric_normalisation_max = colour_metric_normalisation_max
        self.colour_metric_normalisation_power = colour_metric_normalisation_power


    @property
    def displacement_scale_factor(self) -> float:
        return self.universe.trajectory.displacement_scale_factor

    @displacement_scale_factor.setter
    def displacement_scale_factor(self, value: float):
        self.universe.trajectory.displacement_scale_factor = value
        if not self.is_running:
            self.send_frame(self, self.frame_server)

    def set_up_commands(self):
        """Register the playback control commands with the frame server.

        This method maps the VR commands for play, pause, reset, and step to the
        corresponding methods in this class.
        """
        self.frame_server.server.register_command(PLAY_COMMAND_KEY, self.play)
        self.frame_server.server.register_command(PAUSE_COMMAND_KEY, self.pause)
        self.frame_server.server.register_command(RESET_COMMAND_KEY, self.reset)
        # No UI button currently provided for this action in NanoVer-IMD
        self.frame_server.server.register_command(STEP_COMMAND_KEY, self.step)

    @property
    def is_running(self):
        """Boolean indicating if the server is running."""
        return self._run_task is not None and not (self._run_task.cancelled() or self._run_task.done())

    def play(self):
        """Starts or resumes the trajectory playback.

        Cancels any existing playback and starts a new one.
        """
        # First, we have to cancel any existing playback, and start a new one.
        with self._cancel_lock:
            self.cancel_playback(wait=True)
        self.run_playback()

    def step(self):
        """Advances the trajectory by a single frame and then stops."""
        # The lock here ensures only one person can cancel at a time.
        with self._cancel_lock:
            self.cancel_playback(wait=True)
            self._step_one_frame()

    def pause(self):
        """Pause trajectory playback."""
        with self._cancel_lock:
            self.cancel_playback(wait=True)

    def run_playback(self, block=False):
        """Runs the trajectory playback.

        If ``block`` is `False`, then the server will run on a background thread.

        Arguments:
            block: A boolean indicating if the playback should block the main thread.
        """
        if self.is_running:
            raise RuntimeError("The trajectory is already playing on a thread!")
        if block:
            self._run()
        else:
            self._run_task = self.threads.submit(self._run)

    def _run(self):
        """Handles the playback loop.

        This method is called in a background thread to continuously send
        frames at the specified frame rate until playback is cancelled.
        """
        while not self._cancelled:
            self._step_one_frame()
            time.sleep(1 / self.fps)  # Delay sending frames so we hit the desired FPS
        self._cancelled = False

    def _step_one_frame(self):
        """Sends the current frame and advances the frame index."""
        self.send_frame(self, self.frame_server)
        self.frame_index = (self.frame_index + 1) % self.universe.trajectory.n_frames

    def cancel_playback(self, wait=False):
        """Cancels the trajectory playback if it is running.

        Arguments:
            wait: A boolean indicating if the method should wait until playback
                stops before returning.
        """
        if self._run_task is None:
            return

        if self._cancelled:
            return
        self._cancelled = True
        if wait:
            self._run_task.result()
            self._cancelled = False

    def reset(self):
        """Resets the trajectory playback to the first frame."""
        self.frame_index = 0

    def send_topology_frame(self):
        """Sends the topology frame to the NanoVer frame server.

        Converts the MDAnalysis topology to a NanoVer frame and sends it
        to the frame server.
        """
        # Convert the mdanalysis topology to a NanoVer frame
        frame = mdanalysis_to_frame_data(self.universe, topology=True, positions=False)
        self.frame_server.frame_publisher.send_frame(0, frame)

    @staticmethod
    def exponential_normalisation(x, x_min, x_max, p):
        return (x**p - x_min**p) / (x_max**p - x_min**p)

    @staticmethod
    def send_frame(trajectory_player, frame_server):
        """Sends the current frame's particle positions to the NanoVer frame server.

        Arguments:
            trajectory_player: The TrajectoryPlayback object.
            frame_server: The NanoVer frame server to send frame data to.
        """
        index = trajectory_player.frame_index
        # Send the particle positions of the given trajectory index.
        assert 0 <= index < trajectory_player.universe.trajectory.n_frames, f'Frame index not in range [{0},{trajectory_player.universe.trajectory.n_frames - 1}]'
        # Set the target frame (setting the target frame by getting the time step somewhat non-pythonic)
        _ = trajectory_player.universe.trajectory[index]
        frame = mdanalysis_to_frame_data(trajectory_player.universe, topology=False, positions=True)

        # TODO: Sort out order information to ensure that things on the client
        #   side will match up.
        if index > 0:
            norm_displacements = np.linalg.norm(
                trajectory_player.universe.trajectory._displacements[index, :, :], axis=-1)

            norm_displacements = trajectory_player.exponential_normalisation(
                norm_displacements,
                trajectory_player.colour_metric_normalisation_min,
                trajectory_player.colour_metric_normalisation_max,
                trajectory_player.colour_metric_normalisation_power)

            frame.arrays.set("residue.normalised_metric_colour", norm_displacements)

        # A value of one must be added to the frame index to prevent sending
        # the value "zero" which is a special reset command used to delete
        # all stored data on the client side.
        frame_server.frame_publisher.send_frame(index + 1, frame)


