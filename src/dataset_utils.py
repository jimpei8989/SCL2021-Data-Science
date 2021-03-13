def _find_matching(address, target):
    if target == "":
        return -1

    address = address.replace(",", "").split()
    target = target.split()

    for i in range(len(address) - len(target) + 1):
        if all(
            b.startswith(a)
            for a, b in zip(address[i : i + len(target)], target)
        ):
            return i
    return False


def align_to_address(data):
    return data | {
        "poi_index": _find_matching(data["address"], data["poi"]),
        "street_index": _find_matching(data["address"], data["street"]),
    }
