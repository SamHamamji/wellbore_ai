import re

scientific_notation = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
file_regex = r".*ISO_vs({0})vp({1})_MP_dipole\.mat$".format(
    scientific_notation, scientific_notation
)


def get_filter_fn_by_vs_vp(
    max_vs: int | float = float("inf"), max_vp: int | float = float("inf")
):
    def filter_fn(file: str) -> bool:
        match = re.match(file_regex, file)
        if match is None:
            return False

        vs, vp = match.groups()
        if (float(vp) >= max_vp) or (float(vs) >= max_vs):
            return False

        return True

    return filter_fn
