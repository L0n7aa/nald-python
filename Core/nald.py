import argparse
import config


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-M', '--molecules', help='number of molecules')
    parser.add_argument('-A', '--atoms', help='number of atoms')
    parser.add_argument('-T', '--temperature', help='value of the temperature for this simulation')
    args = vars(parser.parse_args())
    print(args)

    config.N_atoms = int(args['atoms'])
    config.N_molecules = int(args['molecules'])
    config.T = float(args['temperature'])

    from Situation.situation import situation
    from LAMMPS.lammps_scripts import main as lmp_scripts
    from Diagonalization.scripts_diag import main as diag
    from Modulus.modulus import compute as mod
    from Plot.plot import plot_moduli

    while True:
        situation()
    
        while True:
            prompt = input("Do you want to regenerate the system? [y/n] : ").strip().lower()
            if prompt in ('y', 'n'):
                break
            print("Incorrect response.")
        
        if prompt == 'n':
            break
        elif prompt == 'y':
            print("Regenerating...")
    lmp_scripts()
    diag()
    mod()

    while True:
        prompt = input("Do you want to plot the moduli? [y/n] : ").strip().lower()
        if prompt in ('y', 'n'):
            break
        print("Incorrect response.")
    if prompt == 'y':
        plot_moduli()