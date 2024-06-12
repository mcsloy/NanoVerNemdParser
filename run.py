import MDAnalysis
from parser import NemdDisplacementFrame
from trajectory_generator import TrajectoryGenerator
from nemd_playback import TrajectoryPlayback
import numpy as np


def determine_scaling_bounds(trajectory: TrajectoryGenerator):
    """Identifies the minimum and maximum displacements.

    This is intended to be used to work out the upper and lower scaling bounds
    for normalising the displacement to within the bounds of [0, 1] as is
    required for colouring the various residues.

    Arguments:
        trajectory: the `TrajectoryGenerator` entity from which the atomic
            displacement information can be sourced.

    Returns:
        minimum_displacement: The minimum displacement value.
        maximum_displacement: The maximum displacement value.
    """
    displacements = np.linalg.norm(trajectory._displacements, axis=-1)
    return np.min(displacements), np.max(displacements)


if __name__ == "__main__":
    """Example parser for sending NEMD displacement data to NanoVer-IMD.

    This script uses a custom file parser entity, `NemdDisplacementFrame`, to load
    in and represent NEMD displacement data. It can also convert a sequence of such
    files into a trajectory amenable to representation by `MDAnalysis`.

    The `MDAnalysis` based trajectories may then be hosted by NanoVer using a
    traditional server structure, an interface for which is provided by the
    `TrajectoryPlayback` class.

    Usage:
        Specify the location of the base structural file using the variable
        `reference_structure_file_path`. The directory in which the displacement
        files are stored may be provided via `displacement_file_directory_path`.
        The displacement files will then be auto-loaded by the helper function
         `auto_load_displacement_frames`. However, this may be done manually
         if the file names do not match the initially agreed upon file naming
         scheme, i.e. "average_xyz_displacement_<TIME>ps" where "<TIME>" is some
         integer specifying the time in picoseconds.
         
         Other trajectory and visualisation options can be controlled by the
         supplementary variables provided in the code shown below. Note that
         documentation for these options is provided in the comments rather
         than within this docstring.  

    Warning:
        This will load all frame data into memory via an `MDAnalysis.MemoryReader`
        instance. This may result in significant memory usage. If this becomes an
        issue, then `numpy.memmap` arrays may need to be implemented to lighten
        the memory footprint.
    """
    from nanover.app import NanoverImdClient

    # ╔════════════════════╗
    # ║      Settings      ║
    # ╚════════════════════╝
    #
    # Path to the reference structure files
    reference_structure_file_path = r"protein_file_path.pdb"
    # Path to the directory in which the NEMD displacment files are located
    displacement_file_directory_path = r"path/to/displacement/file/directory"

    # Trajectory frames shown per second.
    fps = 15

    # Should MDAnalysis attempt to infer atomic bonds?
    # The NanoVer client requires a bond list to utilise the "cartoon" renderer.
    # Some structure files include bond data, but not all do. If bond data is
    # absent from the structural file, set this flag to `True` to instruct
    # MDAnalysis to infer the bonds. Note that for larger systems, this process
    # can be extremely time-consuming. Therefore, it is recommended to perform
    # this inference once and then save the file with the bond data for future
    # use using the command `universe.atoms.write('protein_with_bonds.pdb', bonds='all')`.
    should_compute_bonds = False

    # Specifies what type of renderer is to be used to visualise the structure
    # by NanoVer clients. By default, a cartoon based renderer is used.
    renderer = "cartoon extended"

    # Given that some displacements can be too small to easily see, it is
    # sometimes best to scale them to magnify the visual effects. Atomic
    # displacements will be scaled by the factor specified below before being
    # added to the reference frame. Note that this scaling can be done
    # dynamically while the simulation is running via either the
    # `TrajectoryGenerator` or `TrajectoryPlayback` entities.
    displacement_scale_factor = 1.0

    # If the "nemd" renderer is used, each residue will be coloured according
    # to its displacement. The following variables set the initial range of the
    # colour map. Ideally, the minimum and maximum values should correspond to
    # the minimum and maximum displacements. It is important to note that the
    # displacement values used for sampling the colour map do not include the
    # `displacement_scale_factor`. That is to say if the maximum displacement
    # present is 0.8 then `colour_metric_normalisation_max` should be set to
    # `0.8` and not `0.8 * displacement_scale_factor`. If These values are not
    # assigned then an estimation will be made later on in this script.
    colour_metric_normalisation_min = None
    colour_metric_normalisation_max = None

    # ╔════════════════════╗
    # ║   Initialisation   ║
    # ╚════════════════════╝
    #
    # Construct a `Universe` entity and load in the structure file. Note that
    # if the file does not contain information about bonds, then MDAnalysis
    # must be instructed to calculate the bonds by setting the `guess_bonds`
    # flag.
    universe = MDAnalysis.Universe(reference_structure_file_path, guess_bonds=False)

    # Load in the displacement data into a series of `NemdDisplacementFrame` instances
    displacement_frames = NemdDisplacementFrame.auto_load_displacement_frames(
        displacement_file_directory_path)

    # Note that displacement data can also be loaded manually like so:
    # displacement_frames = list(map(NemdDisplacementFrame.load, [path_one, path_two, ...]))

    # Construct the trajectory entity, and assign it to the universe object
    trajectory = TrajectoryGenerator.from_nemd_displacement_frames(displacement_frames, universe)
    universe.trajectory = trajectory

    # If not scaling bounds were supplied then just set them to the maximum and
    # minimum displacement values.
    if not colour_metric_normalisation_min or not colour_metric_normalisation_max:
        min_x, max_x = determine_scaling_bounds(trajectory)
        if not colour_metric_normalisation_min:
            colour_metric_normalisation_min = min_x
        if not colour_metric_normalisation_max:
            colour_metric_normalisation_max = max_x

    # ╔════════════════════╗
    # ║      Playback      ║
    # ╚════════════════════╝
    #
    # Construct the trajectory playback entity
    trajectory_player = TrajectoryPlayback(
        universe, fps=fps,
        colour_metric_normalisation_min=colour_metric_normalisation_min,
        colour_metric_normalisation_max=colour_metric_normalisation_max)

    # Publish the topology data
    trajectory_player.send_topology_frame()
    # Initiate playback
    trajectory_player.play()

    # ╔════════════════════╗
    # ║      Visuals       ║
    # ╚════════════════════╝
    #
    # Enable cartoon rendering mode, or whatever was specified in the settings section.
    client = NanoverImdClient.autoconnect()
    client.subscribe_multiplayer()
    client.subscribe_to_frames()
    root_selection = client.root_selection
    root_selection.renderer = renderer
    root_selection.flush_changes()

    # Scale the displacements by the following factor to make them more evident
    trajectory_player.displacement_scale_factor = displacement_scale_factor

    # ╔════════════════════╗
    # ║         Run        ║
    # ╚════════════════════╝
    #
    # Do not terminate the python thread, and therefore the server, until the user
    # expressly indicates that it safe to do so via depression of the return key.
    prompt = input("> Press the return key to terminate the server...")
    trajectory_player.pause()
    trajectory_player.frame_server.close()
    exit()
