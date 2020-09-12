from nas_lib.data_darts.data_darts import DataSetDarts


def build_open_search_space_dataset(search_spaces):
    if search_spaces == 'darts':
        return DataSetDarts()
    else:
        raise ValueError("This architecture datasets does not support!")