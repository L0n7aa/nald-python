import numpy as np
import matplotlib.pyplot as plt


# TODO use path 
def load_data(path):
    molecules = np.load(path)          
    n_molecules, n_atoms, _ = molecules.shape
    return molecules, n_molecules, n_atoms


# TODO use cfg 
def save_data(molecules, n_molecules, n_atoms, atom_path):
    #  CSV export  
    ids_molecule = np.repeat(np.arange(n_molecules), n_atoms)
    ids_atom = np.tile(np.arange(n_atoms), n_molecules)
    data = np.column_stack([ids_molecule, ids_atom, molecules.reshape(-1, 3)])

    np.savetxt(atom_path, data, delimiter=",",
            header="molecule,atom,x,y,z", comments="",
            fmt=["%d", "%d", "%.6f", "%.6f", "%.6f"])


# Visualization 
# TODO : upgrade this function for creating multiple type of graphs
#        move this function after calculations
#        move this function in plot directory
def visualization(molecules, n_molecules, image_path):
    colors = plt.cm.hsv(np.linspace(0, 1, n_molecules, endpoint=False))
    mini = molecules.reshape(-1, 3).min(axis=0)
    maxi = molecules.reshape(-1, 3).max(axis=0)

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection="3d")   

    for molecule, c in zip(molecules, colors):
        ax.plot(molecule[:, 0], molecule[:, 1], molecule[:, 2], color=c, linewidth=0.5)

    ax.set_title("3D view")
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
    ax.set_xlim(mini[0], maxi[0]); ax.set_ylim(mini[1], maxi[1]); ax.set_zlim(mini[2], maxi[2])
    ax.set_box_aspect(maxi - mini)               

    plt.tight_layout()
    plt.savefig(image_path, dpi=150)
    print("Written image : visualization.png")
    plt.close(fig)