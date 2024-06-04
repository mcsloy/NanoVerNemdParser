import MDAnalysis
from parser import NemdDisplacementFrame
from trajectory_generator import TrajectoryGenerator
from nemd_playback import TrajectoryPlayback


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


    # Should MDAnalysis attempt to infer atomic bonds?
    # The NanoVer client requires a bond list to utilise the "cartoon" renderer.
    # Some structure files include bond data, but not all do. If bond data is
    # absent from the structural file, set this flag to `True` to instruct
    # MDAnalysis to infer the bonds. Note that for larger systems, this process
    # can be extremely time-consuming. Therefore, it is recommended to perform
    # this inference once and then save the file with the bond data for future
    # use using the command `universe.atoms.write('protein_with_bonds.pdb', bonds='all')`.
    should_compute_bonds = False

    # Given that some displacements can be too small to easily see, it is
    # sometimes best to scale them to magnify the visual effects. Atomic
    # displacements will be scaled by the factor specified below before being
    # added to the reference frame. Note that this scaling can be done
    # dynamically while the simulation is running via either the
    # `TrajectoryGenerator` or `TrajectoryPlayback` entities.
    displacement_scale_factor = 4.0

    # Specifies what type of renderer is to be used to visualise the structure
    # by NanoVer clients.
    renderer = "cartoon"

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

    # ╔════════════════════╗
    # ║      Playback      ║
    # ╚════════════════════╝
    #
    # Construct the trajectory playback entity
    trajectory_player = TrajectoryPlayback(universe, fps=30)

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
