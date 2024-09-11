import time
from os.path import splitext
from typing import Optional
import numpy as np
from concurrent import futures
from threading import RLock
from nanover.app import NanoverFrameApplication
from nanover.mdanalysis import mdanalysis_to_frame_data
from nanover.omni.record import record_from_server
from nanover.trajectory import FrameData
from nanover.trajectory.frame_server import \
    PLAY_COMMAND_KEY, RESET_COMMAND_KEY, STEP_COMMAND_KEY, PAUSE_COMMAND_KEY, STEP_BACK_COMMAND_KEY

from matplotlib import colormaps


class TrajectoryPlayback:
    """MDAnalysis based trajectory playback.

    A class to manage the playback of a molecular dynamics trajectory using MDAnalysis
    and NanoVer.

    This class provides methods to control the playback, such as playing, pausing,
    stepping through frames, and resetting the trajectory. It also handles the communication
    with the NanoVer frame server to update the particle positions in the VR environment.

    This playback instance is specifically designed to allow residues to be coloured and
    scaled as a function of their displacement.

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
        residue_scale_from: The size of each residue will be scaled by a value
            between ``residue_scale_from`` and ``residue_scale_to`` depending
            on the normalised displacement metric. The ``residue_scale_from``
            attribute specifies the minimum scaling bounds. A value of one
            here would indicate that residues with no displacement will have
            an "unmodified" size.
        residue_scale_to: The maximum size to which residues may be scaled.
        colour_map_name: The name of a MatPlotLib gradient to use when colouring
            the residues. Available colour maps are listed in the dictionary
            `matplotlib.colormaps`. For more information on available colour
            maps see matplotlib.org/stable/users/explain/colors/colormaps.html.
            This will default to `"viridis"`.
        record_to_file: File path specifying the location at which a cording of
            the associated NanoVer session should be stored. If no file path
            is provided, then no recording will be made. This will default to
            `None`, i.e. no recording.

    """

    def __init__(
            self, universe, frame_server=None, fps=60,
            colour_metric_normalisation_min: float = 0.0,
            colour_metric_normalisation_max: float = 1.0,
            colour_metric_normalisation_power: float = 1.0,
            residue_scale_from: float = 1.0,
            residue_scale_to: float = 5.0,
            colour_map_name: Optional[str] = "viridis",
            record_to_file: Optional[str] = None):
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
        residue_scale_from: The size of each residue will be scaled by a value
            between ``residue_scale_from`` and ``residue_scale_to`` depending
            on the normalised displacement metric. The ``residue_scale_from``
            attribute specifies the minimum scaling bounds. A value of one
            here would indicate that residues with no displacement will have
            an "unmodified" size.
        residue_scale_to: The maximum size to which residues may be scaled.
        colour_map_name: The name of a MatPlotLib gradient to use when colouring
            the residues. Available colour maps are listed in the dictionary
            `matplotlib.colormaps`. For more information on available colour
            maps see matplotlib.org/stable/users/explain/colors/colormaps.html.
            This will default to `"viridis"`.
        record_to_file: File path specifying the location at which a cording of
            the associated NanoVer session should be stored. If no file path
            is provided, then no recording will be made. This will default to
            `None`, i.e. no recording.
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

        self._colour_map_name = colour_map_name

        self._residue_scale_from = residue_scale_from
        self._residue_scale_to = residue_scale_to

        self._send_meta_data = True

        if colour_map_name and colour_map_name not in colormaps:
            raise KeyError(
                f"The supplied name \"{colour_map_name}\" does not correspond "
                f"to a valid matplotlib colour map.")

        self._record_to_file = record_to_file

        if record_to_file:
            self.record()

    @property
    def displacement_scale_factor(self) -> float:
        return self.universe.trajectory.displacement_scale_factor

    @displacement_scale_factor.setter
    def displacement_scale_factor(self, value: float):
        self.universe.trajectory.displacement_scale_factor = value
        if not self.is_running:
            self.send_frame()

    @property
    def residue_scale_from(self) -> float:
        return self._residue_scale_from

    @residue_scale_from.setter
    def residue_scale_from(self, value: float):
        self._residue_scale_from = value
        self._update_meta_data()

    @property
    def residue_scale_to(self) -> float:
        return self._residue_scale_to

    @residue_scale_to.setter
    def residue_scale_to(self, value: float):
        self._residue_scale_to = value
        self._update_meta_data()

    @property
    def colour_map_name(self) -> str:
        return self._colour_map_name

    @colour_map_name.setter
    def colour_map_name(self, name: str):
        if name not in colormaps:
            raise KeyError(
                f"The supplied name \"{name}\" does not correspond to a valid "
                f"matplotlib colour map.")
        self._colour_map_name = name
        self._update_meta_data()

    @property
    def record_to_file(self) -> str:
        return self._record_to_file

    def _update_meta_data(self):
        self._send_meta_data = True
        if not self.is_running:
            self.send_frame()

    def set_up_commands(self):
        """Register the playback control commands with the frame server.

        This method maps the VR commands for play, pause, reset, and step to the
        corresponding methods in this class.
        """
        self.frame_server.server.register_command(PLAY_COMMAND_KEY, self.play)
        self.frame_server.server.register_command(PAUSE_COMMAND_KEY, self.pause)
        self.frame_server.server.register_command(RESET_COMMAND_KEY, self.reset)
        self.frame_server.server.register_command(STEP_COMMAND_KEY, self.step)
        self.frame_server.server.register_command(STEP_BACK_COMMAND_KEY, self.step_back)

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

    def step_back(self):
        """Steps trajectory back by a single frame and then stops."""
        # The lock here ensures only one person can cancel at a time.
        with self._cancel_lock:
            self.cancel_playback(wait=True)
            self.frame_index = (self.frame_index - 1) % self.universe.trajectory.n_frames
            self.send_frame()

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
        self.send_frame()
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
        self._add_matplotlib_gradient_to_frame(frame)
        self.frame_server.frame_publisher.send_frame(0, frame)

    @staticmethod
    def exponential_normalisation(x, x_min, x_max, p):
        return (x**p - x_min**p) / (x_max**p - x_min**p)

    def send_frame(self):
        """Sends the current frame's particle positions to the NanoVer frame server."""
        index = self.frame_index

        # Send the particle positions of the given trajectory index.

        if not (0 <= index < self.universe.trajectory.n_frames):
            raise ValueError(f'Frame index not in range [{0}, {self.universe.trajectory.n_frames - 1}]')

        # Set the target frame (setting the target frame by getting the time step somewhat non-pythonic)
        _ = self.universe.trajectory[index]
        frame = mdanalysis_to_frame_data(self.universe, topology=False, positions=True)

        if index > 0:
            norm_displacements = np.linalg.norm(
                self.universe.trajectory._displacements[index, :, :], axis=-1)

            norm_displacements = self.exponential_normalisation(
                norm_displacements,
                self.colour_metric_normalisation_min,
                self.colour_metric_normalisation_max,
                self.colour_metric_normalisation_power)

            frame.arrays.set("residue.normalised_metric_c", norm_displacements)


        if self._send_meta_data:
            frame.values.set("residue.scale_from", self.residue_scale_from)
            frame.values.set("residue.scale_to", self.residue_scale_to)
            self._add_matplotlib_gradient_to_frame(frame)
            self._send_meta_data = False

        # A value of one must be added to the frame index to prevent sending
        # the value "zero" which is a special reset command used to delete
        # all stored data on the client side.
        self.frame_server.frame_publisher.send_frame(index + 1, frame)

    def _add_matplotlib_gradient_to_frame(self, frame: FrameData):
        """Append colour gradient array data to specified frame.

        This function will retrieve the matplotib gradient entity assigned to
        the name specified by `colour_map_name`. It will then sample it, and
        store the resulting rgba data in a flattened array. This array will
        then be added to the provide frame data object.

        Arguments:
            frame: The frame data object within which the resulting gradient
                colour array should be stored.

        """
        if self.colour_map_name:
            colour_map = colormaps[self._colour_map_name]
            colour_map_array = np.array([i for j in range(8) for i in colour_map(j/7)])
            frame.arrays.set("residue.colour_gradient", colour_map_array)

    def record(self):
        # Note, using "localhost" will not necessarily continue to work when
        # and if server authentication is implemented.
        record_from_server(
            f"localhost:{self.frame_server.port}",
            f"{self._record_to_file}.traj",
            f"{self._record_to_file}.state")


