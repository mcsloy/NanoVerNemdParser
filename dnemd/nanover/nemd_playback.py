import time
from typing import Optional
import numpy as np
from concurrent import futures
from threading import RLock
from matplotlib import colormaps
from MDAnalysis import Universe

from nanover.app import NanoverFrameApplication
from nanover.app import NanoverImdClient
from nanover.mdanalysis import mdanalysis_to_frame_data
from nanover.omni.record import record_from_server
from nanover.trajectory import FrameData
from nanover.trajectory.frame_server import \
    PLAY_COMMAND_KEY, RESET_COMMAND_KEY, STEP_COMMAND_KEY, PAUSE_COMMAND_KEY, STEP_BACK_COMMAND_KEY

from .generators import DNemdTrajectoryGenerator


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
        displacement_normalisation_lower_bound: Minimum value for the normalisation
            of the displacement metric. See the notes section for more information.
        displacement_normalisation_upper_bound: Maximum value for the normalisation
            of the displacement metric.
        displacement_normalisation_exponent: Power used for non-linear
            normalisation of the displacement metric.

    Properties:
        residue_scale_minimum: The size of each residue will be scaled by a value
            between ``residue_scale_minimum`` and ``residue_scale_maximum`` depending
            on the normalised displacement metric. The ``residue_scale_minimum``
            attribute specifies the minimum scaling bounds. A value of one
            here would indicate that residues with no displacement will have
            an "unmodified" size.
        residue_scale_maximum: The maximum size to which residues may be scaled.
        colour_map_name: The name of a MatPlotLib gradient to use when colouring
            the residues. Available colour maps are listed in the dictionary
            `matplotlib.colormaps`. For more information on available colour
            maps see matplotlib.org/stable/users/explain/colors/colormaps.html.
            This will default to `"viridis"`.
        record_to_file: File path specifying the location at which a cording of
            the associated NanoVer session should be stored. If no file path
            is provided, then no recording will be made. This will default to
            `None`, i.e. no recording.
        alpha: Transparency override value. By default, both colour & transparency
            are controlled by the supplied colour map. Most matplotlib colour maps,
            however, are fully opaque. If an alpha value is specified, it will
            globally override the transparency set by the gradient. For example,
            setting ``alpha=0.5`` will render the structure at 50% transparency.
            This is useful for improving clarity in dense systems or when multiple
            molecules are overlaid. The default value is `None`, meaning the
            transparency from the colour map will be used.


    Notes:
        During playback, residue displacement distances are normalised to the range [0, 1],
        producing a *normalised displacement metric*. This metric controls visual
        properties such as residue colour and size. Residues with smaller displacements
        will be assigned the visual attributes associated with the lower bound (e.g.
        default size and colour), while residues with larger displacements will scale up
        accordingly, reflecting their higher displacement.  The bounds for this
        normalisation are determined by `displacement_normalisation_lower_bound` and
        `displacement_normalisation_upper_bound`, which are typically set to the minimum
        and maximum displacement distances, respectively. Furthermore, the
        `displacement_normalisation_exponent` can be used to add a degree of non-linearity
        to the normalised displacement metric.
    """

    def __init__(
            self, universe: Universe, frame_server=None, fps: int = 60,
            displacement_normalisation_lower_bound: Optional[float] = None,
            displacement_normalisation_upper_bound: Optional[float] = None,
            displacement_normalisation_exponent: float = 1.0,
            residue_scale_minimum: float = 1.0,
            residue_scale_maximum: float = 5.0,
            colour_map_name: Optional[str] = "viridis",
            alpha: Optional[float] = None,
            record_to_file: Optional[str] = None):
        """Initialises the TrajectoryPlayback object.

        Sets up the MDAnalysis universe, frame server, and frames per second (fps)
        for playback.

        Arguments:
            universe: The MDAnalysis universe object containing the trajectory data.
            frame_server: The NanoVer frame server to send frame data to. If None, a new
                server is created.
            fps: The number of frames per second for the trajectory playback.
            displacement_normalisation_lower_bound: Minimum value for the normalisation
                of the displacement metric. If no lower bound is specified then it will
                default to the minimum displacement.
            displacement_normalisation_upper_bound: Maximum value for the normalisation
                of the displacement metric. If no upper bound is specified then it will
                default to the maximum displacement.
            displacement_normalisation_exponent: Power used for non-linear
                normalisation of the displacement metric.
            residue_scale_minimum: The size of each residue will be scaled by a value
                between ``residue_scale_minimum`` and ``residue_scale_maximum`` depending
                on the normalised displacement metric. The ``residue_scale_minimum``
                attribute specifies the minimum scaling bounds. A value of one
                here would indicate that residues with no displacement will have
                an "unmodified" size.
            residue_scale_maximum: The maximum size to which residues may be scaled.
            colour_map_name: The name of a MatPlotLib gradient to use when colouring
                the residues. Available colour maps are listed in the dictionary
                `matplotlib.colormaps`. For more information on available colour
                maps see matplotlib.org/stable/users/explain/colors/colormaps.html.
                This will default to `"viridis"`.
            alpha: Overrides the transparency value. By default, both colour and
                transparency are controlled by the supplied colour map. Most
                matplotlib colour maps, however, are fully opaque. If an alpha value
                is specified, it will globally override the transparency set by the
                gradient. For example, setting ``alpha=0.5`` will render the structure
                at 50% transparency. This is useful for improving clarity in dense
                systems or when multiple molecules are overlaid. The default value
                is `None`, meaning the transparency from the colour map will be used.
            record_to_file: File path specifying the location at which a cording of
                the associated NanoVer session should be stored. If no file path
                is provided, then no recording will be made. This will default to
                `None`, i.e. no recording.

        Notes:
            During playback, residue displacement distances are normalised to the range [0, 1],
            producing a *normalised displacement metric*. This metric controls visual
            properties such as residue colour and size. Residues with smaller displacements
            will be assigned the visual attributes associated with the lower bound (e.g.
            default size and colour), while residues with larger displacements will scale up
            accordingly, reflecting their higher displacement.  The bounds for this
            normalisation are determined by `displacement_normalisation_lower_bound` and
            `displacement_normalisation_upper_bound`, which are typically set to the minimum
            and maximum displacement distances, respectively. Furthermore, the
            `displacement_normalisation_exponent` can be used to add a degree of non-linearity
            to the normalised displacement metric.

        """
        self.universe = universe

        self.frame_server = frame_server

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

        # If no bounds are provided then they will be set to the min/max displacement by the
        # `send_frame` method.
        self.displacement_normalisation_lower_bound = displacement_normalisation_lower_bound
        self.displacement_normalisation_upper_bound = displacement_normalisation_upper_bound
        self.displacement_normalisation_exponent = displacement_normalisation_exponent

        self._colour_map_name = colour_map_name
        self._alpha = alpha

        self._residue_scale_minimum = residue_scale_minimum
        self._residue_scale_maximum = residue_scale_maximum

        self._send_meta_data = True

        # Used to track if topology has been sent at the start of the simulation
        self.__initial_topology_has_been_sent = False

        self._record_to_file = record_to_file

        self.__root_selection = None
        self.__client = None

        if record_to_file:
            self.record()

        if colour_map_name and colour_map_name not in colormaps:
            raise KeyError(
                f"The supplied name \"{colour_map_name}\" does not correspond "
                f"to a valid matplotlib colour map.")

        if not isinstance(universe.trajectory, DNemdTrajectoryGenerator):
            raise TypeError("Trajectory object must be a `SimpleGenerator` instance.")

        # Ensure that the alpha value is a known good type.
        if not isinstance(alpha, (float, type(None))):
            raise TypeError(
                f"The alpha value may be a float or `None`, but \"{alpha}\" was provided.")

        # If a float value is provided then confirm that it is within bounds.
        if isinstance(alpha, float) and not (0.0 <= alpha <= 1.0):
            raise ValueError(
                f"Provided alpha value, {alpha}, is out of bounds; permitted domain [0, 1]")


    @property
    def displacement_scale_factor(self) -> float:
        return self.universe.trajectory.displacement_scale_factor

    @displacement_scale_factor.setter
    def displacement_scale_factor(self, value: float):
        self.universe.trajectory.displacement_scale_factor = value
        if not self.is_running:
            self.send_frame()

    @property
    def residue_scale_minimum(self) -> float:
        return self._residue_scale_minimum

    @residue_scale_minimum.setter
    def residue_scale_minimum(self, value: float):
        self._residue_scale_minimum = value
        self._update_meta_data()

    @property
    def residue_scale_maximum(self) -> float:
        return self._residue_scale_maximum

    @residue_scale_maximum.setter
    def residue_scale_maximum(self, value: float):
        self._residue_scale_maximum = value
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
    def alpha(self):
        """Global transparency override value."""
        return self._alpha

    @alpha.setter
    def alpha(self, value: float | None):
        # Ensure that the alpha value is a known good type.
        if not isinstance(value, (float, type(None))):
            raise TypeError(
                f"The alpha value may be a float or `None`, but \"{value}\" was provided.")

        # If a float value is provided then confirm that it is within bounds.
        if isinstance(value, float) and not (0.0 <= value <= 1.0):
            raise ValueError(
                f"Provided alpha value, {value}, is out of bounds; permitted domain [0, 1]")

        # If everything is good, then set the alpha value.
        self._alpha = value
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

        if self.frame_server is None:
            self.frame_server = NanoverFrameApplication.basic_server(port=0)

        if self.is_running:
            raise RuntimeError("The trajectory is already playing on a thread!")

        if not self.__initial_topology_has_been_sent:
            self.send_topology_frame()
            self.__initial_topology_has_been_sent = True

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

    def close(self):
        """Close down the server."""
        self.pause()
        self.cancel_playback(True)
        self.frame_server.close()
        self.frame_server = None
        self.__initial_topology_has_been_sent = False
        self._send_meta_data = True
        self.__root_selection = True
        del self.__client
        self.__client = None

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

        trajectory: DNemdTrajectoryGenerator = self.universe.trajectory

        if not isinstance(trajectory, DNemdTrajectoryGenerator):
            raise TypeError("Trajectory object must be a `SimpleGenerator` instance.")

        # Send the particle positions of the given trajectory index.
        if not (0 <= index < self.universe.trajectory.n_frames):
            raise ValueError(f'Frame index not in range [{0}, {self.universe.trajectory.n_frames - 1}]')

        # Set the target frame (setting the target frame by getting the time step somewhat non-pythonic)
        _ = self.universe.trajectory[index]
        frame = mdanalysis_to_frame_data(self.universe, topology=False, positions=True)

        if self.displacement_normalisation_lower_bound is None:
            self.displacement_normalisation_lower_bound = trajectory.minimum_displacement_distance

        if self.displacement_normalisation_upper_bound is None:
            self.displacement_normalisation_upper_bound = trajectory.maximum_displacement_distance

        norm_displacements = self.exponential_normalisation(
            trajectory.get_displacement_norms(),
            self.displacement_normalisation_lower_bound,
            self.displacement_normalisation_upper_bound,
            self.displacement_normalisation_exponent)

        frame.arrays.set("residue.normalised_metric_c", norm_displacements)

        if self._send_meta_data:
            frame.values.set("residue.scale_from", self.residue_scale_minimum)
            frame.values.set("residue.scale_to", self.residue_scale_maximum)
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
            if self._alpha is not None:
                for i in range(3, len(colour_map_array), 4):
                    colour_map_array[i] = self._alpha
            frame.arrays.set("residue.colour_gradient", colour_map_array)

    def set_global_renderer(self, renderer: str):
        """Apply renderer to root selection.

        This method will apply the specified renderer to the root selection,
        i.e. all atoms in the system.

        Arguments:
            renderer: Name of the renderer to be applied to the root selection.
        """

        # Ensure that a frame server exists, otherwise setting the renderer via this
        # method has no meaning.
        if self.frame_server is None:
            self.frame_server = NanoverFrameApplication.basic_server(port=0)

        if self.__client is None:
            self.__client = NanoverImdClient.autoconnect()
            self.__client.subscribe_multiplayer()
            self.__client.subscribe_to_frames()
            self.__root_selection = self.__client.root_selection

        self.__root_selection.renderer = renderer
        self.__root_selection.flush_changes()

    def record(self):
        # Note, using "localhost" will not necessarily continue to work when
        # and if server authentication is implemented.
        record_from_server(
            f"localhost:{self.frame_server.port}",
            f"{self._record_to_file}.traj",
            f"{self._record_to_file}.state")
