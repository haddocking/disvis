import itertools

import numpy as np

# Support both fftw and numpy
try:
    import pyfftw

    PYFFTW = True
except ImportError:
    PYFFTW = False

from .volume import Volume
from ._extensions import fill_restraint_space, fill_restraint_space_add


class InteractionSpace(object):
    """Calculates the interaction space between two macromolecules."""

    def __init__(self, space, rcore, rsurface, lcore, max_clash=200, min_inter=300):

        self.space = space
        self.rcore = rcore
        self.rsurface = rsurface
        self.lcore = lcore
        self.max_clash = max_clash
        self.min_inter = min_inter
        self._shape = self.rcore.shape
        self._ft_shape = list(self._shape)[:-1] + [self._shape[-1] // 2 + 1]
        # Subtract/add a small amount to account for later FFT round-off errors
        # TODO test correct behaviour
        self._max_clash_grid = np.floor(self.max_clash / self.rcore.voxelspacing ** 3) + 0.1
        self._min_inter_grid = np.ceil(self.min_inter / self.rcore.voxelspacing ** 3) - 0.1

        # Allocate space for arrays
        # Real arrays
        array_names = 'tmp clashspace interspace'.split()
        for arr in array_names:
            if PYFFTW:
                setattr(self, '_' + arr,
                        pyfftw.zeros_aligned(self._shape, dtype=np.float64))
            else:
                setattr(self, '_' + arr,
                        np.zeros(self._shape, dtype=np.float64))
        array_names = 'not_clashing interacting'.split()
        for arr in array_names:
            setattr(self, "_" + arr, np.zeros(self._shape, dtype=np.int32))

        # Complex arrays
        array_names = 'rcore rsurface lcore lcore_conj tmp'.split()
        for arr in array_names:
            if PYFFTW:
                setattr(self, '_ft_' + arr,
                        pyfftw.zeros_aligned(self._ft_shape, dtype=np.complex128))
            else:
                setattr(self, '_ft_' + arr,
                        np.zeros(self._ft_shape, dtype=np.complex128))
        # Setup FFT
        if PYFFTW:
            self._rfftn = pyfftw.builders.rfftn(self.rcore.array)
            self._irfftn = pyfftw.builders.irfftn(self._ft_rcore, s=self._shape)

        else:
            def rfftn(a, b):
                b[:] = np.fft.rfftn(a)

            def irfftn(a, b):
                b[:] = np.fft.irfftn(a, s=self._shape)

            self._rfftn = rfftn
            self._irfftn = irfftn

        # Perform initial calculations
        self._rfftn(self.rcore.array, self._ft_rcore)
        self._rfftn(self.rsurface.array, self._ft_rsurface)

    def __call__(self):
        """Calculate the interaction space."""

        self._rfftn(self.lcore.array, self._ft_lcore)
        np.conjugate(self._ft_lcore, self._ft_lcore_conj)
        # Calculate the clashing volume
        np.multiply(self._ft_lcore_conj, self._ft_rcore, self._ft_tmp)
        self._irfftn(self._ft_tmp, self._clashspace)
        # np.round(self._clashspace, out=self._clashspace)
        # Calculate the interaction volume
        np.multiply(self._ft_lcore_conj, self._ft_rsurface, self._ft_tmp)
        self._irfftn(self._ft_tmp, self._interspace)
        # np.round(self._interspace, out=self._interspace)
        # Determine complexes with too many clashes and too few interactions
        np.less_equal(self._clashspace, self._max_clash_grid,
                      self._not_clashing)
        np.greater_equal(self._interspace, self._min_inter_grid,
                         self._interacting)
        np.logical_and(self._not_clashing, self._interacting,
                       self.space.array)


class Restraint(object):
    """An (ambiguous) distance restraint."""

    def __init__(self, rselections, lselections, min_dis, max_dis):
        self.rselections = rselections
        self.lselections = lselections
        self.min = min_dis
        self.max = max_dis


class RestraintSpace(object):
    """Determine consistent restraint space."""

    def __init__(self, space, restraints, ligand_center):
        self.restraints = restraints
        self.space = space
        # Transform the restraint coordinates to grid coordinates
        self._restraints_grid = []
        for restraint in self.restraints:
            rcoor = np.vstack(rsel.coor for rsel in restraint.rselections)
            rcoor = (rcoor.reshape(-1, 3) - self.space.origin) / self.space.voxelspacing

            lcoor = np.vstack(lsel.coor for lsel in restraint.lselections)
            lcoor = (lcoor.reshape(-1, 3) - ligand_center) / self.space.voxelspacing
            min_dis = restraint.min / self.space.voxelspacing
            max_dis = restraint.max / self.space.voxelspacing
            restraint_grid = Restraint(rcoor, lcoor, min_dis, max_dis)
            self._restraints_grid.append(restraint_grid)
        self.nrestraints = len(self.restraints)

        # Number of permutations of a multi-set
        self.npermutations = 1 << self.nrestraints
        self.indices = np.arange(self.npermutations, dtype=np.int32)
        self.consistent_restraints = np.zeros(self.npermutations, dtype=np.int8)
        mask = np.zeros(self.npermutations, dtype=np.int32)
        for n in xrange(self.nrestraints):
            value = 1 << n
            # Project the consistent restraint out
            np.bitwise_and(self.indices, value, mask)
            self.consistent_restraints[mask > 0] += 1

    def __call__(self, rotmat):
        self.space.array.fill(0)
        for n, restraint in enumerate(self._restraints_grid):
            lcoor = np.ascontiguousarray(np.dot(restraint.lselections, rotmat.T))
            value = 1 << n
            fill_restraint_space(
                restraint.rselections, lcoor,
                restraint.min, restraint.max, value, self.space.array)


class AccessibleInteractionSpace(object):
    def __init__(self, space, interaction_space, restraint_space):
        self.space = space
        self.interaction_space = interaction_space
        self.restraint_space = restraint_space
        self.max_consistent = Volume.zeros_like(space)
        self.nrestraints = restraint_space.nrestraints
        self.npermutations = self.restraint_space.npermutations
        self._indices = self.restraint_space.indices
        self._consistent_restraints = self.restraint_space.consistent_restraints
        self._consistent_permutations = np.zeros(self.npermutations, dtype=np.float64)
        self.consistent_space = Volume.zeros_like(self.space)

    def __call__(self, weight=1):
        np.multiply(self.interaction_space.space.array,
                    self.restraint_space.space.array, self.space.array)
        self.consistent_space.array[:] = self._consistent_restraints[self.space.array]
        np.maximum(self.max_consistent.array,
                   self.consistent_space.array,
                   self.max_consistent.array)
        counts = np.bincount(self.space.array.ravel(), minlength=self.npermutations)[1:]
        self._consistent_permutations[1:] += counts * weight
        self._consistent_permutations[0] += weight * (
            self.interaction_space.space.array.sum() - counts.sum())

    def consistent_complexes(self, exact=False):
        """
        Return the number of complexes consistent with a number of restraints.
        """
        out = np.zeros(self.nrestraints + 1, dtype=np.float64)
        for n in xrange(self.nrestraints + 1):
            mask = self._consistent_restraints == n
            out[n] = self._consistent_permutations[mask].sum()
        if not exact:
            out = np.cumsum(out[::-1])[::-1]
        return out

    def consistent_matrix(self, exact=False):
        out = np.zeros((self.nrestraints + 1, self.nrestraints), dtype=np.float64)
        for nconsistent in xrange(self.nrestraints + 1):
            mask1 = self._consistent_restraints == nconsistent
            for restraint_index in xrange(self.nrestraints):
                restraint_bit = 1 << restraint_index
                mask2 = np.bitwise_and(self._indices, restraint_bit) > 0
                out[nconsistent, restraint_index] += self._consistent_permutations[
                    np.logical_and(mask1, mask2)].sum()
        if not exact:
            out = np.cumsum(out[::-1], axis=0)[::-1]
        return out

    def violation_matrix(self, exact=False):
        out = self.consistent_complexes(exact=exact).reshape(-1, 1) - self.consistent_matrix(exact=exact)
        return out


class ResidueInteractionSpace(object):
    def __init__(self, space, receptor, ligand, accessible_interaction_space,
                 interaction_radius=3):
        self.space = space
        self.ligand = ligand
        self.receptor = receptor
        self.accessible_interaction_space = accessible_interaction_space
        self._ligand_coor = np.ascontiguousarray(
            (self.ligand.coor - self.ligand.center).T / self.space.voxelspacing)
        self._ligand_coor_rot = np.zeros_like(self._ligand_coor)
        self._receptor_coor = np.ascontiguousarray((self.receptor.coor -
                                                    self.space.origin) / self.space.voxelspacing)
        # Create distance restraints between each each residue of the ligand
        # and all of the receptor and each residue of the receptor with all of
        # the ligand. The distance restraints need to have views on the ligand
        # and receptor coordinate for efficiency.
        self._restraints_ligand = []
        self._restraints_receptor = []
        # Build up the restraints for each ligand residue
        unique_residues, unique_indices = np.unique(
            self.ligand.data['resi'], return_indices=True)
        unique_indices = unique_indices.tolist() + [None]
        max_dis = 5.0 / self.space.voxelspacing
        for start, end in izip(unique_indices[:-1], unique_indices[1:]):
            coor = self._ligand_coor[start:end]
            restraint = Restraint(self._receptor_coor, coor, 0, max_dis)
            self._restraints_ligand.append(restraint)
        # Build up the restraints for each receptor residue
        unique_residues, unique_indices = np.unique(
            self.receptor.data['resi'], return_indices=True)
        unique_indices = unique_indices.tolist() + [None]
        for start, end in izip(unique_indices[:-1], unique_indices[1:]):
            coor = self._receptor_coor[start:end]
            restraint = Restraint(self._receptor_coor, coor, 0, max_dis)
            self._restraints_ligand.append(restraint)

    def __call__(self, rotmat, weight=1):
        # Rotate all atoms and thus views of the ligand's coordinates
        np.dot(self._ligand_coor, rotmat.T, out=self._ligand_coor_rot)
        for n, restraint in enumerate(self._ligand_restraints):
            fill_restraint_space_add(
                restraint.rselections, restraint.lselections,
                restraint.min, restraint.max, self.space.array)
            # TODO Count interactions for consistent complexes
            self.space.array.fill(0)

        for n, restraint in enumerate(self._receptor_restraints):
            fill_restraint_space_add(
                restraint.rselections, restraint.lselections,
                restraint.min, restraint.max, self.space.array)
            # TODO Count interactions for consistent complexes
            self.space.array.fill(0)


class OccupancySpace(object):
    def __init__(self, interaction_space, accessible_interaction_space, nconsistent=None):
        self.interaction_space = interaction_space
        self.accessible_interaction_space = accessible_interaction_space
        self.nconsistent = nconsistent
        if self.nconsistent is None:
            nrestraints = self.accessible_interaction_space.nrestraints
            minconsistent = max(1, nrestraints - 2)
            self.nconsistent = range(minconsistent, nrestraints + 1)
        if PYFFTW:
            self._consistent = pyfftw.zeros_aligned(
                self.interaction_space.space.array.shape, dtype=np.float64)
        else:
            self._consistent = np.zeros(
                self.interaction_space.space.array.shape, dtype=np.float64)
        self._tmp = interaction_space._tmp
        self._ft_tmp = interaction_space._ft_tmp
        self._rfftn = self.interaction_space._rfftn
        self._irfftn = self.interaction_space._irfftn
        self.spaces = []
        for n in self.nconsistent:
            self.spaces.append(
                Volume.zeros_like(self.interaction_space.space, dtype=np.float64)
            )

    def __call__(self, weight=1):
        for n, space in itertools.izip(self.nconsistent, self.spaces):
            np.greater_equal(self.accessible_interaction_space.consistent_space.array,
                             n, out=self._consistent)
            self._rfftn(self._consistent, self._ft_tmp)
            self._ft_tmp *= self.interaction_space._ft_lcore
            self._irfftn(self._ft_tmp, self._tmp)
            self._tmp *= weight
            space.array += self._tmp
