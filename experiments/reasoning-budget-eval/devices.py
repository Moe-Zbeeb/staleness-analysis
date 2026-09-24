def select_devices(allocation, by_index, expected):
    selected = []
    for item in allocation.split(','):
        item = item.strip()
        if item.startswith('GPU-'):
            selected.append(item)
        elif '-' in item:
            low, high = map(int, item.split('-'))
            selected.extend(by_index[str(i)] for i in range(low, high + 1))
        else:
            selected.append(by_index[item])
    assert len(selected) == expected and len(set(selected)) == expected
    assert set(selected) <= set(by_index.values())
    return selected
