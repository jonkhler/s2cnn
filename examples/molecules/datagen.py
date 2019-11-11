import argparse
import numpy as np
import scipy.io as spio
from scipy.spatial import distance as spdist
import joblib
import lie_learn.spaces.S2 as S2


MAX_NUM_ATOMS_PER_MOLECULE = 23
NUM_ATOM_TYPES = 5


def get_raw_data(path):
    """ load data from matlab file """
    raw = spio.loadmat(path)
    coordinates = raw["R"]
    charges = raw["Z"]
    energies = raw["T"]
    strat_ids = raw["P"]
    return coordinates, charges, energies, strat_ids


def get_projection_grid(bandwidth, grid_type="Driscoll-Healy"):
    theta, phi = S2.meshgrid(b=bandwidth, grid_type=grid_type)
    x_ = np.sin(theta) * np.cos(phi)
    y_ = np.sin(theta) * np.sin(phi)
    z_ = np.cos(theta)
    return np.array((x_, y_, z_))


def compute_features_for_molecule(unique_charges, mol_coords, mol_charges,
                                  atom_grids, grid_bandwidth, min_atom_dist=1):
    # output features
    num_atoms = len(mol_coords)
    mol_features = np.ndarray((len(unique_charges), num_atoms,
                               2*grid_bandwidth, 2*grid_bandwidth))

    # for each possible interacting atom type
    # compute one feature map of interaction
    for i, probe_charge in enumerate(unique_charges):
        features = np.sum(compute_coloumb_forces(atom_grids, probe_charge,
                                                 mol_coords, mol_charges,
                                                 min_atom_dist),
                          axis=0)
        mol_features[i] = features.reshape(num_atoms, 2*grid_bandwidth,
                                           2*grid_bandwidth)
    return mol_features


def compute_coloumb_forces(atom_grids, probe_charge, mol_coords, mol_charges,
                           min_atom_dist=1):
    inv_sq_distance = compute_inv_sq_distances(atom_grids, probe_charge,
                                               mol_coords, mol_charges,
                                               min_atom_dist)
    f = inv_sq_distance * probe_charge * mol_charges.reshape(1, -1, 1)
    return f


def compute_inv_sq_distances(atom_grids, probe_charge, mol_coords, mol_charges,
                             min_atom_dist=1):
    num_atoms = len(mol_coords)

    # all positions of atoms having the probe charge
    probe_coords = mol_coords[mol_charges == probe_charge]

    # if there is no atom with the probe charge we are done
    if len(probe_coords) == 0:
        return np.zeros((1, num_atoms, atom_grids.shape[0]//num_atoms))

    # spatial distance between z atoms and grid points
    distances = spdist.cdist(probe_coords, atom_grids).reshape(
        len(probe_coords), num_atoms, -1)

    # for each atom in the molecule set the distance
    # to the grid point around that atom to 0
    nonzero = (distances - distances.min(axis=1).reshape(
        len(probe_coords), 1, -1)) > 0

    distances = nonzero * 1 / distances**2

    return distances


def get_min_distance(coords, charges):
    num_molecules = len(coords)
    distances = []
    for mol_idx in range(num_molecules):
        non_null_atoms = charges[mol_idx] != 0
        atom_coords = coords[mol_idx][non_null_atoms]
        pairwise_atom_distances = spdist.pdist(atom_coords)
        distances.append(pairwise_atom_distances.min())
    # TODO remove unncesseary O(N) traversal by accumulator
    min_distance = np.min(distances)
    return min_distance


def generate_dataset(coordinates, charges, energies, strat_ids,
                     grid_bandwidth):
    num_molecules = len(coordinates)

    data = {}
    data["features"] = {
        "geometry": np.zeros((num_molecules, MAX_NUM_ATOMS_PER_MOLECULE,
                              NUM_ATOM_TYPES, 2*grid_bandwidth,
                              2*grid_bandwidth)),
        "atom_types": np.zeros((num_molecules, MAX_NUM_ATOMS_PER_MOLECULE,)),
        "num_atoms": np.zeros((num_molecules,))
    }
    data["targets"] = energies.T
    data["strats"] = strat_ids

    unique_charges = np.sort(np.unique(charges))

    # compute minimum distance in the data set
    min_atom_dist = get_min_distance(coordinates, charges)

    # get grid for bandwidth
    grid = (get_projection_grid(grid_bandwidth)
            * min_atom_dist).reshape(3, -1).T

    for mol_idx in range(num_molecules):

        print("\rprocessing molecule {0}/{1}".format(
            mol_idx+1, num_molecules), end="")

        non_null_atoms = charges[mol_idx] != 0
        num_non_null_atoms = sum(non_null_atoms)

        # get position and types of non NULL atoms
        mol_coords = coordinates[mol_idx][non_null_atoms]
        mol_charges = charges[mol_idx][non_null_atoms]

        # copy grid around each atom
        atom_grids = (np.stack((grid,)*mol_coords.shape[0], 0)
                      + mol_coords[:, np.newaxis, :]).reshape(-1, 3)

        mol_features = compute_features_for_molecule(
            unique_charges[1:], mol_coords, mol_charges, atom_grids,
            grid_bandwidth, min_atom_dist)

        # transpose features to right shape
        axes = np.arange(len(mol_features.shape))
        axes[0:2] = (1, 0)
        mol_features = np.transpose(mol_features, axes)

        # copy to data dict
        data["features"]["geometry"][
            mol_idx, :num_non_null_atoms, ...] = mol_features
        data["features"]["atom_types"][mol_idx][
            :num_non_null_atoms] = mol_charges
        data["features"]["num_atoms"] = non_null_atoms

    print("")
    return data


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--bandwidth",
                        help="the bandwidth of the S2 signal",
                        type=int,
                        default=10,
                        required=False)
    parser.add_argument("--data_file",
                        help="file for saving the data output (.gz file)",
                        type=str,
                        default="qm7.mat",
                        required=False)
    parser.add_argument("--output_file",
                        help="file for saving the data output (.gz file)",
                        type=str,
                        default="data.joblib",
                        required=False)

    args = parser.parse_args()

    raw_data = get_raw_data(args.data_file)

    data = generate_dataset(*raw_data, args.bandwidth)

    print("save to file")
    joblib.dump(data, args.output_file)


if __name__ == '__main__':
    main()
