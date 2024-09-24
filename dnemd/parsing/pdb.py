import os
from typing import Self, Optional
from abc import ABC, abstractmethod
import tempfile
import re
import numpy as np
from numpy.typing import NDArray
from MDAnalysis.topology.PDBParser import PDBParser
from MDAnalysis.core.topology import Topology


class PdbRecord(ABC):
    """Abstract base class for Protein Data Bank (PDB) formatted fixed width files."""

    @classmethod
    @abstractmethod
    def from_string(cls, string: str) -> Self:
        """Instantiate a protein data bank record from string."""
        pass

    @abstractmethod
    def __str__(self) -> str:
        """Serialise protein data bank record to string."""
        pass

    @abstractmethod
    def copy(self) -> Self:
        """Copy protein data bank record."""
        pass


class PdbFile(PdbRecord, ABC):
    """Abstract base class for Protein Data Bank (PDB) files."""
    def to_mdanalysis_topology(self) -> Topology:
        """Generate a `MDAnalysis.Topology` instance from the pdb file.


        This method instantiates a new `MDAnalysis.Topology` objected based on
        the data specified by the protein data band file.

        Returns:
            MDAnalysis Topology object based on the current record.
        """

        # A temporary file is created to store the PDB data for MDAnalysis. The
        # file is created outside a context manager to prevent it from being
        # deleted too early.
        temp_file = tempfile.NamedTemporaryFile(delete=False)

        # Ensure the temporary file is cleaned up after parsing.
        try:
            # Write the PDB data to the temporary file.
            temp_file.write(str(self).encode("utf-8"))
            # Ensure the data is written to disk before parsing.
            temp_file.flush()
            # Parse the PDB file to create an MDAnalysis topology object.
            topology = PDBParser(temp_file.name).parse()

        finally:
            # Close and delete the temporary file after use.
            temp_file.close()
            os.remove(temp_file.name)

        return topology


class PdbAtomRecord(PdbRecord):
    """Protein Data Bank (PDB) atom record.

    Atomic record storing the orthogonal coordinates for atoms in standard
    residue, such as amino and nucleic acids.

    Attributes:
        atom_serial_number: The unique serial number assigned to each atom.
            It is used to differentiate between different atoms within the
            same structure. This value is extracted from columns 7-11.
        atom_name: The name of the atom, typically following IUPAC nomenclature.
            Whitespace offsets for single-character element names are handled
            internally, so names should be provided without such offsets.
            Extracted from columns 13-16.
        residue_name: The name of the residue to which the atom belongs.
            Residue names are standardised, such as three-letter codes for
            amino acids or nucleotides. This value is extracted from columns
            18-20.
        chain_identifier: A single-character identifier assigned to the chain
            in which the atom is located. This value is used to distinguish
            atoms in different chains and is extracted from column 22.
        residue_sequence_number: The sequence number of the residue to which
            the atom belongs. It is used to order the residues within a chain.
            This value is extracted from columns 23-26.
        orthogonal_coordinates: A numpy array storing the x, y, and z coordinates
            of the atom in Cartesian space. These coordinates are used to define
            the atom's position within the structure and are extracted from
            columns 31-38 (x), 39-46 (y), and 47-54 (z).
        occupancy: A float value representing the occupancy of the atom, indicating
            the fraction of the atom present at the given position. This value is
            extracted from columns 55-60.
        temperature_factor: A float value representing the B-factor (or temperature
            factor), which measures the displacement of the atom due to thermal
            motion. This value is extracted from columns 61-66.
        elemental_symbol: A two-character string representing the chemical element
            of the atom (e.g., C for carbon, O for oxygen). This value is extracted
            from columns 77-78.
    """
    def __init__(self,
                 atom_serial_number: int,
                 atom_name: str,
                 residue_name: str,
                 chain_identifier: str,
                 residue_sequence_number: int,
                 orthogonal_coordinates: NDArray[float],
                 occupancy: float,
                 temperature_factor: float,
                 elemental_symbol: str):
        """Atom record constructor method.


        Arguments:
            atom_serial_number: The unique serial number assigned to each atom.
                It is used to differentiate between different atoms within the
                same structure. This value is extracted from columns 7-11.
            atom_name: The name of the atom, typically following IUPAC nomenclature.
                Whitespace offsets for single-character element names are handled
                internally, so names should be provided without such offsets.
                Extracted from columns 13-16.
            residue_name: The name of the residue to which the atom belongs.
                Residue names are standardised, such as three-letter codes for
                amino acids or nucleotides. This value is extracted from columns
                18-20.
            chain_identifier: A single-character identifier assigned to the chain
                in which the atom is located. This value is used to distinguish
                atoms in different chains and is extracted from column 22.
            residue_sequence_number: The sequence number of the residue to which
                the atom belongs. It is used to order the residues within a chain.
                This value is extracted from columns 23-26.
            orthogonal_coordinates: A numpy array storing the x, y, and z coordinates
                of the atom in Cartesian space. These coordinates are used to define
                the atom's position within the structure and are extracted from
                columns 31-38 (x), 39-46 (y), and 47-54 (z).
            occupancy: A float value representing the occupancy of the atom, indicating
                the fraction of the atom present at the given position. This value is
                extracted from columns 55-60.
            temperature_factor: A float value representing the B-factor (or temperature
                factor), which measures the displacement of the atom due to thermal
                motion. This value is extracted from columns 61-66.
            elemental_symbol: A two-character string representing the chemical element
                of the atom (e.g., C for carbon, O for oxygen). This value is extracted
                from columns 77-78.
        """

        # Column 0: record type
        # ATOM

        # Column 1:
        self.atom_serial_number = atom_serial_number

        # Column 2:
        self.atom_name = atom_name

        # Column 3:
        self.residue_name = residue_name

        # Column 4:
        self.chain_identifier = chain_identifier

        # Column 5:
        self.residue_sequence_number = residue_sequence_number

        # Column 6, 7, 8:
        self.orthogonal_coordinates = orthogonal_coordinates

        # Column 9:
        self.occupancy = occupancy

        # Column 10:
        self.temperature_factor = temperature_factor

        # Column 11:
        self.elemental_symbol = elemental_symbol

    @classmethod
    def from_string(cls, string: str) -> Self:
        """Instantiate a protein data bank ATOM record from string.

        Arguments:
            string: a string representing a single protein data bank atom record.

        """

        if not string.startswith("ATOM"):
            raise ValueError(
                f"Strings representing pdb atom records are expected to start "
                f"with the the string \"ATOM\". The provided string does not:\n"
                f"\t{string}")

        return cls(
            # Atom serial number
            int(string[6:11]),
            # Atom name
            string[12:16].strip(),
            # Residue name
            string[17:20].strip(),
            # Chain identifier
            string[21],
            # Residue sequence number
            int(string[22:26]),
            # Coordinates
            np.array(string[30:54].split(), dtype=float),
            # Occupancy
            float(string[54:60]),
            # Temperature factor
            float(string[60:66]),
            # Elemental symbol
            string[76:78].strip()
        )

    def __str__(self) -> str:
        """Serialise protein data bank ATOM record to a string.

        Returns:
            record_string: a string representing a serialised instance of a
                protein data bank atom record.
        """
        if len(self.elemental_symbol) == 1 and len(self.atom_name) <= 3:
            padded_name = " " + self.atom_name
        else:
            padded_name = self.atom_name

        return (
            f"ATOM  "
            f"{self.atom_serial_number: >5} "
            f"{padded_name: <4} "
            f"{self.residue_name: >2} "
            f"{self.chain_identifier}"
            f"{self.residue_sequence_number: >4}    "
            f"{self.orthogonal_coordinates[0]: >8.3f}"
            f"{self.orthogonal_coordinates[1]: >8.3f}"
            f"{self.orthogonal_coordinates[2]: >8.3f}"
            f"{self.occupancy: >6.2f}"
            f"{self.temperature_factor: >6.2f}          "
            f"{self.elemental_symbol: >2}")

    def copy(self) -> Self:
        """Create a copy of the protein data bank atom record.

        Returns:
            atom_record_copy: a copy of the protein data bank atom record.
        """
        return PdbAtomRecord(
            self.atom_serial_number,
            self.atom_name,
            self.residue_name,
            self.chain_identifier,
            self.residue_sequence_number,
            self.orthogonal_coordinates.copy(),
            self.occupancy,
            self.temperature_factor,
            self.elemental_symbol
        )


class PdbCryst1(PdbRecord):
    """Protein Data Bank (PDB) CRYST1 record.

    The CRYST1 record defines the unit cell parameters and the space group
    for a crystallographic structure. It provides the dimensions of the
    crystallographic lattice and the symmetry information.

    Attributes:
        lattice_vector: A numpy array storing the lengths of the unit cell's
            lattice vectors (a, b, and c) in angstroms. These values are
            extracted from columns 7-15 (a), 16-24 (b), and 25-33 (c).
        angles: A numpy array storing the angles (alpha, beta, and gamma)
            between the unit cell's lattice vectors. These angles are extracted
            from columns 34-40 (alpha), 41-47 (beta), and 48-54 (gamma).
        space_group: A string representing the space group symbol for the
            crystallographic structure. This value is extracted from
            columns 56-66.
        z_value: An integer representing the number of asymmetric units per
            unit cell. This value is extracted from columns 67-70.
    """

    def __init__(self,
                 lattice_vector: NDArray[float],
                 angles: NDArray[float],
                 space_group: str,
                 z_value: int):
        """CRYST1 record constructor method.

        Arguments:
            lattice_vector: A numpy array containing the lengths of the unit
                cell's lattice vectors (a, b, and c), in angstroms.
                These are extracted from columns 7-15 (a), 16-24 (b),
                and 25-33 (c).
            angles: A numpy array containing the unit cell angles (alpha,
                beta, and gamma), in degrees. These are extracted from
                columns 34-40 (alpha), 41-47 (beta), and 48-54 (gamma).
            space_group: A string representing the space group symbol for
                the crystallographic structure. This value is extracted from
                columns 56-66.
            z_value: An integer representing the number of asymmetric units
                per unit cell. This value is extracted from columns 67-70.
        """
        self.lattice_vector = lattice_vector
        self.angles = angles
        self.space_group = space_group
        self.z_value = z_value

    @classmethod
    def from_string(cls, string: str) -> Self:
        """Instantiate a protein data bank CRYST1 record from a string.

        Arguments:
            string: A string representing a single CRYST1 record in the
                Protein Data Bank format.

        Returns:
            A new instance of the `PdbCryst1` class.
        """
        return cls(
            # Lattice vector
            np.array(string[6:33].split(), dtype=float),
            # Cell angles
            np.array(string[33:54].split(), dtype=float),
            # Space group
            string[55:66].strip(),
            # Z-value
            int(string[66:70]),
        )

    def __str__(self) -> str:
        """Serialise a PDB CRYST1 record to a string.

        Returns:
            record_string: A string representing a serialised instance
                of a PDB CRYST1 record.
        """
        return (f"CRYST1"
                f"{self.lattice_vector[0]: >9.3f}"
                f"{self.lattice_vector[1]: >9.3f}"
                f"{self.lattice_vector[2]: >9.3f}"
                f"{self.angles[0]: >7.2f}"
                f"{self.angles[1]: >7.2f}"
                f"{self.angles[2]: >7.2f} "
                f"{self.space_group: <11}"
                f"{self.z_value: >4}")

    def copy(self) -> Self:
        """Create a copy of the PDB CRYST1 record.

        Returns:
            cryst1_record_copy: A copy of the PDB CRYST1 record.
        """
        return PdbCryst1(
            self.lattice_vector.copy(),
            self.angles.copy(),
            self.space_group,
            self.z_value,
        )


class SimpleAtomsOnlyPdbFile(PdbFile):
    """Simple atoms only protein data bank (PDB) file.

    This class is used to read, write, and manipulate a basic protein data bank
    file comprised of a single system represented by a single `CRYST1` record
    and multiple `ATOM` records.

    Attributes:
        crystal_record: a protein data bank `CRYST1` record defining the unit
            cell parameters and space group.
        atom_records: a list of protein data bank `ATOM` records, defining the
            atoms present in the system.
    """
    def __init__(self, crystal_record: PdbCryst1, atom_records: list[PdbAtomRecord]):
        """Constructor method for simple, atom only, protein data bank files.

        Attributes:
            crystal_record: a protein data bank `CRYST1` record defining the unit
                cell parameters and space group.
            atom_records: a list of protein data bank `ATOM` records, defining the
                atoms present in the system.
        """
        self.crystal_record = crystal_record
        self.atom_records = atom_records

    @classmethod
    def from_string(cls, string: str) -> Self:
        """Instantiate a simple atoms only PDB file instance from a string.

        Arguments:
            string: A string representing a single protein data bank file.

        Returns:
            A new instance of the `SimpleAtomsOnlyPdbFile` class.
        """
        cryst1 = PdbCryst1.from_string(
            re.search(r"^CRYST1.+$", string, re.MULTILINE).group(0))

        atoms = [PdbAtomRecord.from_string(i)
                 for i in re.findall(r"^ATOM.+$", string, re.MULTILINE)]

        return cls(cryst1, atoms)

    def __str__(self) -> str:
        """Serialise a simple, atoms only, PDB file to a string.

        Returns:
            record_string: A string representing a serialised instance
                of a simple, atoms only, PDB file.
        """
        crystal_record_string = str(self.crystal_record)
        atom_records_string = "\n".join([str(record) for record in self.atom_records])
        string = crystal_record_string + "\n" + atom_records_string + "\n" + "ENDMDL" + "\n"
        return string

    def copy(self) -> Self:
        """Create a copy of the simple, atoms only, PDB file.

        Returns:
            simple_pdb_file_copy: A copy of the simple, atoms only,
                PDB file `SimpleAtomsOnlyPdbFile` instance.
        """
        return SimpleAtomsOnlyPdbFile(
            self.crystal_record.copy(),
            [i.copy() for i in self.atom_records]
        )


def double_pdb(pdb_file: SimpleAtomsOnlyPdbFile) -> PdbFile:

    atom_records = [i.copy() for i in pdb_file.atom_records]

    max_atomic_index = max(i.atom_serial_number for i in atom_records)
    max_residue_index = max(i.residue_sequence_number for i in atom_records)

    offset_atomic_records = []

    for atom_record in pdb_file.atom_records:
        offset_atom_record = atom_record.copy()
        offset_atom_record.atom_serial_number += max_atomic_index
        atom_record.residue_sequence_number += max_residue_index

        offset_atomic_records.append(atom_record)

    return SimpleAtomsOnlyPdbFile(
        pdb_file.crystal_record.copy(),
        [*atom_records, *offset_atomic_records])


def load_pdb_file_as_doubled_mdanalysis_topology(file_path: str):
    with open(file_path, "r") as file:
        pdb_file = SimpleAtomsOnlyPdbFile.from_string(file.read())

    doubled_pdb_file = double_pdb(pdb_file)
    return doubled_pdb_file.to_mdanalysis_topology()
