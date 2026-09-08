import math
import re

import os.path as op


def extract_docking_score(docking_outfile):
    """
    Parse docking score (kcal/mol) from GNINA / Vina output file.
    Edit patterns below if your log format differs.
    """
    if not op.isfile(docking_outfile):
        raise FileNotFoundError(docking_outfile)

    with open(docking_outfile, "r", encoding="utf-8", errors="replace") as fh:
        text = fh.read()

    patterns = [
        r"Affinity:\s*([-\d.]+)",
        r"REMARK VINA RESULT:\s*([-\d.]+)",
        r"minimizedAffinity\s+([-\d.]+)",
    ]
    for pat in patterns:
        m = re.search(pat, text)
        if m:
            return float(m.group(1))

    raise ValueError("Could not parse docking score from %s" % docking_outfile)
