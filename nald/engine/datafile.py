"""Lecture de l'en-tête d'un data file LAMMPS et vérifications de cohérence avec la config."""
import re

SECTIONS = ("Masses", "Atoms", "Velocities", "Bonds", "Angles", "Dihedrals", "Impropers",
            "Pair Coeffs", "PairIJ Coeffs", "Bond Coeffs", "Angle Coeffs",
            "Dihedral Coeffs", "Improper Coeffs")
_SECTION_RE = re.compile(r"^(" + "|".join(SECTIONS) + r")\s*(?:#\s*(.*?))?\s*$")
_COUNT_RE = re.compile(r"^(\d+)\s+(atoms|bonds|angles|dihedrals|impropers)\b")
_TYPE_RE = re.compile(r"^(\d+)\s+(atom|bond|angle|dihedral|improper)\s+types\b")
_BOX_RE = re.compile(r"^(\S+)\s+(\S+)\s+([xyz])lo\s+[xyz]hi")
_TILT_RE = re.compile(r"^(\S+)\s+(\S+)\s+(\S+)\s+xy\s+xz\s+yz")

STYLE_OF_SECTION = {"Pair Coeffs": "pair_style", "Bond Coeffs": "bond_style",
                    "Angle Coeffs": "angle_style", "Dihedral Coeffs": "dihedral_style",
                    "Improper Coeffs": "improper_style"}
STYLE_OF_KIND = {"bond": "bond_style", "angle": "angle_style",
                 "dihedral": "dihedral_style", "improper": "improper_style"}


def read_header(path):
    """Renvoie {'counts','types','box','tilt','sections'}; sections = {nom: commentaire ou None}."""
    info = {"counts": {}, "types": {}, "box": {}, "tilt": None, "sections": {}}
    in_body = False
    with open(path) as f:
        for i, raw in enumerate(f):
            line = raw.strip()
            if i == 0 or not line:                    # 1re ligne = titre libre
                continue
            if line[0].isalpha():
                m = _SECTION_RE.match(line)
                if m:
                    in_body = True
                    info["sections"][m.group(1)] = m.group(2)
                continue
            if in_body:
                continue
            h = line.split("#", 1)[0].strip()
            for regex, store in ((_COUNT_RE, "counts"), (_TYPE_RE, "types")):
                m = regex.match(h)
                if m:
                    info[store][m.group(2)] = int(m.group(1))
            m = _BOX_RE.match(h)
            if m:
                info["box"][m.group(3)] = (float(m.group(1)), float(m.group(2)))
            m = _TILT_RE.match(h)
            if m:
                info["tilt"] = tuple(float(m.group(k)) for k in (1, 2, 3))
    return info


def describe(hdr):
    c, t = hdr["counts"], hdr["types"]
    n = c.get("atoms", 0)
    lines = [f"  atoms: {n or '?'}   (dim = 3N = {3 * n})",
             "  types: " + ", ".join(f"{k}={v}" for k, v in t.items()),
             "  sections: " + ", ".join(k + (f" [{v}]" if v else "") for k, v in hdr["sections"].items())]
    if hdr["box"]:
        lengths = [hdr["box"][a][1] - hdr["box"][a][0] for a in "xyz" if a in hdr["box"]]
        lines.append("  box lengths: " + " x ".join(f"{L:.4g}" for L in lengths))
    return "\n".join(lines)


def check_consistency(hdr, system, channels):
    """Compare la config au data file. Renvoie une liste d'avertissements (jamais d'exception)."""
    w = []
    declared = {}
    for line in system.styles:
        tok = line.split()
        if tok[0].endswith("_style") and len(tok) > 1:
            declared[tok[0]] = tok[1]
    coeff_cmds = {l.split()[0] for l in system.coefficients}

    atoms_comment = hdr["sections"].get("Atoms")
    if atoms_comment and atoms_comment.split()[0] != system.atom_style:
        w.append(f"atom_style: le data file indique '{atoms_comment.split()[0]}', "
                 f"la config déclare '{system.atom_style}'.")

    for section, key in STYLE_OF_SECTION.items():
        if section not in hdr["sections"]:
            continue
        comment = hdr["sections"][section]
        if key not in declared:
            w.append(f"le data file contient '{section}' mais aucun '{key}' n'est déclaré dans forcefield.styles.")
        elif comment and "hybrid" not in comment and declared[key] != comment.strip():
            w.append(f"{key}: le data file a été écrit avec '{comment.strip()}', "
                     f"la config déclare '{declared[key]}'.")

    for kind, key in STYLE_OF_KIND.items():
        n = hdr["types"].get(kind, 0)
        if n and key not in declared:
            w.append(f"{n} {kind} types dans le data file mais aucun '{key}' dans forcefield.styles.")
        if n and f"{kind.capitalize()} Coeffs" not in hdr["sections"] and f"{kind}_coeff" not in coeff_cmds:
            w.append(f"{n} {kind} types mais ni section '{kind.capitalize()} Coeffs' ni '{kind}_coeff': "
                     "coefficients manquants.")
    if hdr["types"].get("atom"):
        if "Pair Coeffs" not in hdr["sections"] and "PairIJ Coeffs" not in hdr["sections"] \
                and "pair_coeff" not in coeff_cmds:
            w.append("ni 'Pair Coeffs' dans le data file ni 'pair_coeff' dans forcefield.coefficients.")
        if "Masses" not in hdr["sections"] and "mass" not in coeff_cmds:
            w.append("ni section 'Masses' dans le data file ni 'mass' dans forcefield.coefficients.")

    if "coul/long" in declared.get("pair_style", "") and "kspace_style" not in {l.split()[0] for l in system.styles}:
        w.append("pair_style coul/long sans kspace_style dans forcefield.styles.")

    tilt = hdr["tilt"]
    if tilt and any(abs(x) > 0 for x in tilt) and any(c in channels for c in ("xx", "yy", "zz")):
        w.append("boîte initialement triclinique: les canaux xx/yy/zz ne remettent pas les tilts "
                 "à l'échelle (non géré pour l'instant).")
    return w
