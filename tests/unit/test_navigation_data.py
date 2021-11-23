import json
from navigation_analytics.navigation_data import *
import os
from datetime import datetime
import pandas as pd
import pickle
import pytest


def parse_dtypes(dtypes_dict: dict):
    result = dict()
    for col, col_type in dtypes_dict.items():
        result[col] = eval(col_type)
    return result


def load_metadata():
    with open(os.environ['CONFIG_PATH']) as fp:
        config_data = json.load(fp)
    return config_data


def load_data(mode: str == 'pickle'):
    metadata = load_metadata()
    if mode == 'pickle':
        with open(os.environ['PICKLE_PATH'], 'rb') as fp:
            input_data = pickle.load(fp)
    else:
        data_type_dict = parse_dtypes(metadata['metadata']['data_types'])
        input_data = pd.read_csv(os.path.join(metadata['data_import']['file_path'],
                                              metadata['data_import']['file_name']),
                                 dtype=data_type_dict,
                                 na_values=metadata['metadata']['na_vector'])
        input_data.timestamp = [datetime.strptime(str(int(item)), metadata['metadata']['date_format'])
                                for item in input_data.timestamp]
    return metadata, input_data


def _test_baseline_click_through_rate():
    metadata, input_data = load_data(mode='pickle')
    data_analyzer = NavigationDataAnalyzer(input_data=input_data,
                                           metadata=metadata)
    # General ctr
    assert pytest.approx(0.388839,
                         data_analyzer.session_analyzer.compute_click_through_rate(),
                         0.00001)
    # ctr for group a
    assert pytest.approx(0.669657,
                         data_analyzer.session_analyzer.compute_click_through_rate(group_id='a'),
                         0.00001)
    # ctr for group b
    assert pytest.approx(0.174762,
                         data_analyzer.session_analyzer.compute_click_through_rate(group_id='b'),
                         0.00001)


def _test_baseline_most_common_result():
    metadata, input_data = load_data(mode='pickle')
    data_analyzer = NavigationDataAnalyzer(input_data=input_data,
                                           metadata=metadata)
    # General ctr
    assert pytest.approx(0.668576,
                         data_analyzer.session_analyzer.compute_search_frequency()[1.0],
                         0.00001)


def _test_baseline_zero_result_result():
    metadata, input_data = load_data(mode='pickle')
    data_analyzer = NavigationDataAnalyzer(input_data=input_data,
                                           metadata=metadata)
    # General zrr
    assert pytest.approx(0.18444,
                         data_analyzer.session_analyzer.compute_zero_result_rate(),
                         0.00001)
    # Group A zrr
    assert pytest.approx(0.18360,
                         data_analyzer.session_analyzer.compute_zero_result_rate(group_id='a'),
                         0.00001)
    # Group B zrr
    assert pytest.approx(0.18617,
                         data_analyzer.session_analyzer.compute_zero_result_rate(group_id='b'),
                         0.00001)


def _test_baseline_session_length():
    metadata, input_data = load_data(mode='pickle')
    data_analyzer = NavigationDataAnalyzer(input_data=input_data,
                                           metadata=metadata)
    session_length_a = data_analyzer.session_analyzer.compute_session_length(group_id='a')
    session_length_b = data_analyzer.session_analyzer.compute_session_length(group_id='b')
    assert pytest.approx(114.0,
                         session_length_a.median(),
                         0.00001)
    assert pytest.approx(0.0,
                         session_length_b.median(),
                         0.00001)


def _test_compute_all_and_save_object():
    metadata, input_data = load_data(mode='pickle')
    data_analyzer = NavigationDataAnalyzer(input_data=input_data,
                                           metadata=metadata,
                                           logger_level=logging.INFO)
    data_analyzer.session_analyzer.compute_session_length()
    data_analyzer.session_analyzer.compute_session_length(group_id='a')
    data_analyzer.session_analyzer.compute_session_length(group_id='b')
    data_analyzer.session_analyzer.compute_click_through_rate()
    data_analyzer.session_analyzer.compute_click_through_rate(group_id='a')
    data_analyzer.session_analyzer.compute_click_through_rate(group_id='b')
    data_analyzer.session_analyzer.compute_search_frequency()
    data_analyzer.session_analyzer.compute_search_frequency(group_id='a')
    data_analyzer.session_analyzer.compute_search_frequency(group_id='b')
    data_analyzer.session_analyzer.compute_zero_result_rate()
    data_analyzer.session_analyzer.compute_zero_result_rate(group_id='a')
    data_analyzer.session_analyzer.compute_zero_result_rate(group_id='b')
    data_analyzer.save()
    data_analyzer.to_excel('debug_model.xlsx')


def test_object_can_be_loaded():
    data_analyzer = NavigationDataAnalyzer.load(filepath=os.environ['PATH_OBJECT'])
    assert len(data_analyzer.session_analyzer.kpi_results.keys()) > 1
