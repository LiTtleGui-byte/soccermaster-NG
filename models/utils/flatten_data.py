def flatten_data(data):
    if isinstance(data, list):
        result = []
        for item in data:
            if isinstance(item, list):
                result.extend(item)
            else:
                result.append(item)
        return result
    else:
        return data if isinstance(data, list) else [data]