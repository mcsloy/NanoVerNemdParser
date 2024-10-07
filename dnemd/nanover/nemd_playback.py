import time
from typing import Optional
import numpy as np
from threading import Thread, Event
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

        # Ensure that the trajectory is pointing at the first frame.
        _ = universe.universe.trajectory[0]

        self._frame_index = 0

        # Initialise a new frame server as needed
        self.frame_server = frame_server if frame_server is not None else NanoverFrameApplication.basic_server(port=0)

        self.fps = fps

        # The thread performing the playback loop will be stored under this
        # attribute whenever it is actively running.
        self._playback_thread = None

        # This event is used to signal to the playback thread that it should stop.
        self.__cancel_event = Event()

        # Ensure that a valid type of trajectory generator is being used.
        if not isinstance(universe.trajectory, DNemdTrajectoryGenerator):
            raise TypeError("Trajectory object must be a `DNemdTrajectoryGenerator` instance.")

        # If no bounds are provided then they will are set to the minimum and
        # maximum displacements respectively.
        self.displacement_normalisation_lower_bound = (
            displacement_normalisation_lower_bound if displacement_normalisation_lower_bound is not None else
            universe.trajectory.minimum_displacement_distance)

        self.displacement_normalisation_upper_bound = (
            displacement_normalisation_upper_bound if displacement_normalisation_upper_bound is not None else
            universe.trajectory.maximum_displacement_distance)

        self.displacement_normalisation_exponent = displacement_normalisation_exponent

        self._residue_scale_minimum = residue_scale_minimum
        self._residue_scale_maximum = residue_scale_maximum

        if colour_map_name and colour_map_name not in colormaps:
            raise KeyError(
                f"The supplied name \"{colour_map_name}\" does not correspond "
                f"to a valid matplotlib colour map.")

        self._colour_map_name = colour_map_name

        # Ensure that the alpha value is a known good type.
        if not isinstance(alpha, (float, type(None))):
            raise TypeError(
                f"The alpha value may be a float or `None`, but \"{alpha}\" was provided.")

        # If a float value is provided then confirm that it is within bounds.
        if isinstance(alpha, float) and not (0.0 <= alpha <= 1.0):
            raise ValueError(
                f"Provided alpha value, {alpha}, is out of bounds; permitted domain [0, 1]")

        self._alpha = alpha

        self._send_meta_data = True  # Send meta-data in the next `send_frame` call?
        self.__server_has_shutdown = False  # Has the frame-server shutdown?

        self.__root_selection = None
        self.__client = None

        # Set up NanoVer session recording if necessary.
        self._record_to_file = record_to_file
        if record_to_file:
            # Note, using "localhost" will not necessarily continue to work when
            # and if server authentication is implemented.
            record_from_server(
                f"localhost:{self.frame_server.port}",
                f"{self._record_to_file}.traj",
                f"{self._record_to_file}.state")

        # Register the playback control commands with the frame server so that
        # playback commands received by the server map to the relevant functions
        # present in this class.
        self.frame_server.server.register_command(PLAY_COMMAND_KEY, self.play)
        self.frame_server.server.register_command(PAUSE_COMMAND_KEY, self.pause)
        self.frame_server.server.register_command(RESET_COMMAND_KEY, self.reset)
        self.frame_server.server.register_command(STEP_COMMAND_KEY, self.step)
        self.frame_server.server.register_command(STEP_BACK_COMMAND_KEY, self.step_back)

        # Send the topology data along with first trajectory frame. This is done
        # so that there is something to see when the frame-server first starts
        # up. This also gets around the problem where the first frame is skipped
        # in the `_playback_loop` method.
        self.send_topology_frame()
        self.send_frame()

    @property
    def frame_index(self) -> int:
        """Index of current trajectory frame."""
        return self._frame_index

    @property
    def displacement_scale_factor(self) -> float:
        """Displacement scale factor."""
        return self.universe.trajectory.displacement_scale_factor

    @displacement_scale_factor.setter
    def displacement_scale_factor(self, value: float):
        self.universe.trajectory.displacement_scale_factor = value
        if not self.is_running:
            self.send_frame()

    @property
    def residue_scale_minimum(self) -> float:
        """Minimum residue scale factor."""
        return self._residue_scale_minimum

    @residue_scale_minimum.setter
    def residue_scale_minimum(self, value: float):
        self._residue_scale_minimum = value
        self._update_meta_data()

    @property
    def residue_scale_maximum(self) -> float:
        """Maximum residue scale factor."""
        return self._residue_scale_maximum

    @residue_scale_maximum.setter
    def residue_scale_maximum(self, value: float):
        self._residue_scale_maximum = value
        self._update_meta_data()

    @property
    def colour_map_name(self) -> str:
        """Colour map name."""
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
        """Path specifying where playback files will be stored."""
        return self._record_to_file

    def _update_meta_data(self):
        """Resend the current frame with the new meta-data."""
        self._send_meta_data = True
        if not self.is_running:
            self.send_frame()

    @property
    def is_running(self):
        """Boolean indicating if the server is actively playing the trajectory."""
        return self._playback_thread is not None and self._playback_thread.is_alive()

    def play(self):
        """Starts or resumes the trajectory playback."""
        if not self.is_running:
            self.run_playback()

    def pause(self):
        """Pause trajectory playback."""

        # Only attempt to pause/cancel playback if there is something to cancel.
        if self.is_running:
            # Set the `__cancel_event` flag so that the `_playback_loop` method
            # will break out of its while loop.
            self.__cancel_event.set()

            # Wait for the loop to break, this should be quick.
            self._playback_thread.join()

            # Reset the `__cancel_event` flag so that the `_playback_loop`
            # method does not immediately terminate the next time it is called.
            self.__cancel_event.clear()

            # Now that the playback thread has been terminated the associated
            # `_playback_thread` attribute may be cleared
            self._playback_thread = None

    def step(self):
        """Advances the trajectory by a single frame and then stops."""
        self.step_to(self._frame_index + 1)

    def step_back(self):
        """Steps trajectory back by a single frame and then stops."""
        self.step_to(self._frame_index - 1)

    def step_to(self, frame_index: int):
        """Steps trajectory to the specified frame and then stops.

        Arguments:
            frame_index: Index of the frame to go to.
        """
        # Halt playback
        self.pause()
        # Update the current frame index attribute
        self._frame_index = frame_index % self.universe.trajectory.n_frames
        # Send the specified frame.
        self.send_frame()

    def run_playback(self, block=False):
        """Runs the trajectory playback.

        If ``block`` is `False`, then the server will run on a background thread.

        Arguments:
            block: A boolean indicating if the playback should block the main thread.
        """

        if self.is_running:
            raise RuntimeError("The trajectory is already playing on a thread!")

        if self.__server_has_shutdown:
            raise RuntimeError(
                "`TrajectoryPlayback` entities cannot be reused following shutdown.")

        if block:
            self._playback_loop()
        else:
            self._playback_thread = Thread(target=self._playback_loop)
            self._playback_thread.start()

    def _playback_loop(self):
        """Handles the playback loop.

        This method is called in a background thread to continuously send
        frames at the specified frame rate until playback is cancelled.
        """

        # During playback the frame index is iteratively advanced and the
        # corresponding frame data sent. It is important that the frame index
        # is advanced at the start of the loop, rather than at the end. This
        # ensures that the `_frame_index` attribute matches up with the currently
        # displayed frame. The downside to this is that the first frame will be
        # skipped over when this method is called for the very first time.
        while not self.__cancel_event.is_set():

            # Advance to the next frame
            self._frame_index = (self._frame_index + 1) % self.universe.trajectory.n_frames

            # Send the frame
            self.send_frame()

            # Sleep until it is time to send the next frame
            time.sleep(1 / self.fps)

    def reset(self):
        """Reset the trajectory playback to the first frame."""
        self._frame_index = 0

    def close(self):
        """Shutdown the NanoVer server."""
        # Stop the playback thread if it is active
        self.pause()

        # Closed down the ghost client used to define selections
        if self.__client is not None:
            self.__client.close()

        # Terminate the frame server
        self.frame_server.close()

        # Ensure that this playback entity cannot be reused.
        self.__server_has_shutdown = True

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

        index = self._frame_index

        trajectory: DNemdTrajectoryGenerator = self.universe.trajectory

        if not isinstance(trajectory, DNemdTrajectoryGenerator):
            raise TypeError("Trajectory object must be a `DNemdTrajectoryGenerator` instance.")

        # Send the particle positions of the given trajectory index.
        if not (0 <= index < self.universe.trajectory.n_frames):
            raise ValueError(f'Frame index not in range [{0}, {self.universe.trajectory.n_frames - 1}]')

        # Set the target frame (setting the target frame by getting the time step somewhat non-pythonic)
        _ = self.universe.trajectory[index]
        frame = mdanalysis_to_frame_data(self.universe, topology=False, positions=True)

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

        if self.__client is None:
            self.__client = NanoverImdClient.autoconnect()
            self.__client.subscribe_multiplayer()
            self.__client.subscribe_to_frames()
            self.__root_selection = self.__client.root_selection

        self.__root_selection.renderer = renderer
        self.__root_selection.flush_changes()

    def __del__(self):
        self.close()

