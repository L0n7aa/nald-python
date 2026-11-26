"""Remplit les scripts de base (templates/*.in) en remplaçant les marqueurs @@NOM@@."""
import os
import re

TEMPLATE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates")
_MARK = re.compile(r"@@([A-Z0-9_]+)@@")


def render(name, **values):
    """Strict: un marqueur sans valeur, ou une valeur sans marqueur (faute de frappe), est une erreur."""
    with open(os.path.join(TEMPLATE_DIR, name)) as f:
        text = f.read()
    needed = set(_MARK.findall(text))
    missing, unused = needed - set(values), set(values) - needed
    if missing or unused:
        raise KeyError(f"template '{name}': marqueurs sans valeur={sorted(missing)}, "
                       f"valeurs inutilisées={sorted(unused)}")
    return _MARK.sub(lambda m: str(values[m.group(1)]), text)
