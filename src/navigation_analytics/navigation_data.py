import logging
import copy
import pickle
import pandas as pd


class BaseClass:
    def __init__(self,
                 input_data: pd.DataFrame,
                 logger: logging.Logger,
                 metadata: dict):
        self.__input_data = input_data
        self.__logger = logger
        self.__metadata = metadata

    @property
    def logger(self):
        return self.__logger

    @property
    def metadata(self):
        return self.__metadata

    @property
    def input_data(self):
        return self.__input_data

    @input_data.setter
    def input_data(self, new_input_data: pd.DataFrame):
        self.__input_data = new_input_data

    @property
    def events_id(self):
        return self.__metadata['metadata']['primary_keys']['events']

    @property
    def session_id(self):
        return self.__metadata['metadata']['primary_keys']['sessions']

    @property
    def page_id(self):
        return self.__metadata['metadata']['primary_keys']['pages']

    @property
    def group_id(self):
        return self.metadata['metadata']['valid_values']['groups']['group_id']

    @property
    def valid_groups(self):
        return self.metadata['metadata']['valid_values']['groups']['valid']

    @property
    def action_id(self):
        return self.metadata['metadata']['valid_values']['actions']['action_id']

    @property
    def valid_actions(self):
        return self.metadata['metadata']['valid_values']['actions']['valid']

    @property
    def search_action(self):
        return self.metadata['metadata']['valid_values']['actions']['search_action']

    @property
    def visit_action(self):
        return self.metadata['metadata']['valid_values']['actions']['visit_action']

    @property
    def timestamp_id(self):
        return self.metadata['metadata']['datetime']

    @property
    def kpi_duration(self):
        return self.metadata['metadata']['valid_values']['kpis']['duration_page']

    @property
    def kpi_position(self):
        return self.metadata['metadata']['valid_values']['kpis']['result_position']

    @property
    def kpi_number_results(self):
        return self.metadata['metadata']['valid_values']['kpis']['number_results']


class DataValidator(BaseClass):
    def __init__(self,
                 logger: logging.Logger,
                 metadata: dict,
                 input_data: pd.DataFrame):
        super().__init__(logger=logger,
                         metadata=metadata,
                         input_data=input_data)
        self.default_pipeline()

    # Pipelines
    def default_pipeline(self):
        self.check_events_are_unique()
        self.check_groups_are_valid()
        self.check_one_group_per_session()

    # Validation Rules
    def check_events_are_unique(self):
        """
        Verifies that event identifier is primary key of input data.
        :return: Validation
        """
        number_rows = self.input_data.shape[0]
        events_id = self.metadata['metadata']['primary_keys']['events']
        number_events = len(self.input_data[events_id].unique())
        if number_rows == number_events:
            self.logger.info(f'Validation - Events are unique: {number_rows} rows and {number_events} events.')
        else:
            self.logger.error(f'Validation - Events are not unique: {number_rows} rows and {number_events} events.')

    def check_groups_are_valid(self):
        """
        Verifies that groups matches with those declared in metadata.
        :return: Validation
        """
        group_id = self.metadata['metadata']['valid_values']['groups']['group_id']
        groups_in_data = list(self.input_data[group_id].unique())
        group_valid_names = list(self.metadata['metadata']['valid_values']['groups']['valid'])
        if set(groups_in_data) == set(group_valid_names):
            self.logger.info(f'Validation - Groups are valid: {", ".join(group_valid_names)}.')
        else:
            self.logger.error(f'Validation - Group names are not valid: '
                              f'Names in data are {", ".join(groups_in_data)}. '
                              f'Names in metadata are {", ".join(group_valid_names)}.')

    def check_one_group_per_session(self):
        """
        Verifies that there's at most one group per session.
        :return: Validation
        """
        group_id = self.metadata['metadata']['valid_values']['groups']['group_id']
        session_id = self.metadata['metadata']['primary_keys']['sessions']
        max_num_groups = self.input_data.groupby(session_id)[group_id].apply(lambda x: len(set(x))).max()
        if max_num_groups == 1:
            self.logger.info(f'Validation - Just one group per session.')
        else:
            self.logger.error(f'Validation - Groups per session is different to one. '
                              f'Maximum number of groups per session detected in data set is: {max_num_groups}')


class SessionAnalyzer(BaseClass):
    def __init__(self,
                 input_data: pd.DataFrame,
                 metadata: dict,
                 logger: logging.Logger):
        super().__init__(logger=logger,
                         metadata=metadata,
                         input_data=input_data)
        self.__results = dict()
        self.__session_data = self.create_session_look_up()
        self.__page_data = self.create_page_look_up()
        self.__page_data_out = self.create_page_look_up_out()
        self.__search_table = self.create_search_table()
        self.__duration_table = self.create_duration_table()

    def filter_session_by_group(self, group_id: str):
        """
        Filter session by group id provided in the input. This is expected to be a recurrent operation.
        :param group_id:
        :return:
        """
        if group_id not in self.valid_groups:
            self.logger.error(f'{group_id} is not a valid group.')
        return self.session_data.loc[self.session_data[self.group_id] == group_id, :]

    # Metrics
    def compute_click_through_rate(self, group_id: str = None):
        """
        This function computes the click through rate, understanding this quantity as the ratio of searches ending up in
        a session landing in a page. Session Attribute.
        :param group_id:
        :return:
        """
        result = None
        if group_id is None:
            key = 'click_through_rate'
            sub_key = 'all'
            # Merging sessions with page ids
            df = copy.deepcopy(self.session_data.merge(self.page_data, on=self.session_id, how='left'))
            # Computing boolean vector: True means session has a visit, False otherwise.
            result = df.groupby(by=self.session_id)[self.action_id].apply(lambda x: self.visit_action in set(x))
        else:
            key = 'click_through_rate'
            sub_key = group_id
            if group_id in self.valid_groups:
                # Filtering sessions by required group.
                filtered_sessions = self.filter_session_by_group(group_id=group_id)
                df = copy.deepcopy(filtered_sessions.merge(self.page_data, on=self.session_id, how='left'))
                result = df.groupby(by='session_id').action.apply(lambda x: 'visitPage' in set(x))
            else:
                self.logger.error(f'{group_id} is not a valid group.')
        # Computing ctr
        ctr = sum(result) / len(result)
        self.logger.info(f'Click Through Rate is equal to: {ctr}')
        # Storing results
        update_result = self.kpi_results
        try:
            update_result[key][key].append(ctr)
            update_result[key]['group'].append(sub_key)
        except KeyError:
            update_result[key] = dict()
            update_result[key][key] = [ctr]
            update_result[key]['group'] = [sub_key]
        self.kpi_results = update_result
        return ctr

    def compute_search_frequency(self,
                                 group_id: str = None,
                                 number_ranking: int = 10):
        """
        Get the most common first result per session. This is a Session Attribute.
        :param number_ranking: Number of results to visualize.
        :param group_id:
        :return:
        """
        if group_id is None:
            key = 'search_frequency'
            sub_key = 'all'
            df_sessions = self.session_data.copy()
        else:
            key = 'search_frequency'
            sub_key = group_id
            df_sessions = self.filter_session_by_group(group_id=group_id)
        df = df_sessions.merge(self.page_data, on=self.session_id, how='left')
        # Merge with duration table to retrieve datestamp data.
        df_all = df.merge(self.duration_table, on=self.page_id, how='left')
        df_all.dropna(inplace=True)
        # Most common first result
        df_all = df_all.groupby('session_id').apply(lambda x:
                                                    x.loc[x[self.timestamp_id] == min(x[self.timestamp_id]),
                                                          [self.kpi_position, self.timestamp_id]])
        # Result
        result = df_all[self.kpi_position].value_counts(normalize=True)[:number_ranking]
        self.logger.info(f'Most common result is {result.index[0]}')
        # Store result
        updated_results = self.kpi_results
        try:
            updated_results[key][key].extend(list(result.values))
            updated_results[key]['position'].extend(list(result.index))
            updated_results[key]['group'].extend([sub_key]*len(result.index))
        except KeyError:
            updated_results[key] = dict()
            updated_results[key][key] = list(result.values)
            updated_results[key]['position'] = list(result.index)
            updated_results[key]['group'] = [sub_key]*len(result.index)
        self.kpi_results = updated_results
        return result

    def compute_zero_result_rate(self,
                                 group_id: str = None):
        """
        Computes the proportion of searches that end up in no results.
        :param group_id:
        :return:
        """
        df = self.search_table.copy()
        # Compute number of searches resulting in found elements.
        df['success'] = [True if item == 0 else False for item in df[self.kpi_number_results]]
        if group_id is None:
            key = 'zero_result_rate'
            sub_key = 'all'
            result = df['success']
        else:
            key = 'zero_result_rate'
            sub_key = group_id
            df_sessions = self.filter_session_by_group(group_id=group_id)
            df_pages = df_sessions.merge(self.page_data, on=self.session_id, how='left')
            df = df.merge(df_pages, on=self.page_id, how='left')
            df.dropna(inplace=True)
            result = df['success']
        # Computing result
        value = sum(result) / len(result)
        self.logger.info(f'Zero result rate is: {value}')
        # Storing result.
        updated_results = self.kpi_results
        try:
            updated_results[key][key].append(value)
            updated_results[key]['group'].append(sub_key)
        except KeyError:
            updated_results[key] = dict()
            updated_results[key][key] = [value]
            updated_results[key]['group'] = [sub_key]
        self.kpi_results = updated_results
        return value

    def compute_session_length(self,
                               group_id: str = None):
        """
        Compute session's length
        :param group_id:
        :return:
        """
        if group_id is None:
            key = 'session_length'
            sub_key = 'all'
            df = self.input_data
        else:
            key = 'session_length'
            sub_key = group_id
            df = self.filter_session_by_group(group_id=group_id)
            df = df.merge(self.input_data, on=self.session_id, how='left')
        # Compute results
        value = df.groupby(self.session_id)[self.timestamp_id].apply(lambda x: (max(x) - min(x)).total_seconds())
        time_value = df.groupby(self.session_id)[self.timestamp_id].min()
        # Store results
        updated_results = self.kpi_results
        try:
            updated_results[key][key].extend(list(value.values))
            updated_results[key]['session_date'].extend(list(time_value.values))
            updated_results[key]['session_id'].extend(list(value.index))
            updated_results[key]['group'].extend([sub_key]*len(value.index))
        except KeyError:
            updated_results[key] = dict()
            updated_results[key][key] = list(value.values)
            updated_results[key]['session_date'] = list(time_value.values)
            updated_results[key]['session_id'] = list(value.index)
            updated_results[key]['group'] = [sub_key]*len(value.index)
        self.kpi_results = updated_results
        return value

    # Instantiation
    def update_data(self):
        self.page_data = self.create_page_look_up()
        self.page_data_out = self.create_page_look_up_out()
        self.session_data = self.create_session_look_up()
        self.duration_table = self.create_duration_table()
        self.search_table = self.create_search_table()

    def create_session_look_up(self):
        return self.input_data[[self.session_id, self.group_id]].drop_duplicates()

    def create_page_look_up_out(self):
        return self.input_data[[self.session_id, self.page_id]].drop_duplicates()

    def create_page_look_up(self):
        return self.input_data[[self.session_id, self.page_id, self.action_id]].drop_duplicates()

    def create_search_table(self):
        """
        Preserves just search results from original dataset.
        :return: Information relevant only to searches
        """
        local_df = self.input_data.copy()
        local_df = local_df.loc[local_df[self.action_id] == self.search_action,
                                [self.events_id, self.timestamp_id, self.page_id, self.kpi_number_results]]
        return local_df

    def create_duration_table(self):
        """
        Preserves just search results from original dataset.
        :return: Information relevant only to searches
        """
        local_df = self.input_data.copy()
        local_df = local_df.loc[local_df[self.action_id] != self.search_action,
                                [self.timestamp_id,
                                 self.page_id,
                                 self.kpi_position,
                                 self.kpi_duration]]
        # Remove redundant information on position and duration
        local_df = local_df.groupby(self.page_id).max()
        no_duration_info = local_df[self.kpi_duration].isna()
        no_position_info = local_df[self.kpi_position].isna()
        self.logger.warning(f'{no_position_info.sum()} NA values for {self.kpi_position}.')
        self.logger.warning(f'{no_duration_info.sum()} NA values for {self.kpi_duration}.')
        # Remove those observations where position of results do not exist while there is duration
        no_position_but_duration = [(2 * item[1] - item[0]) != 2 for item in zip(no_duration_info, no_position_info)]
        position_but_duration = [(2 * item[1] - item[0]) == 2 for item in zip(no_duration_info, no_position_info)]
        kpi_results = self.kpi_results
        kpi_results['invalid_results'] = local_df.loc[position_but_duration, :].copy()
        self.kpi_results = kpi_results
        self.logger.warning(f'{sum([not item for item in no_position_but_duration])} '
                            f'NA values for position with duration.')
        local_df = local_df.loc[no_position_but_duration, :]
        # The rest of cases fill 0
        local_df.fillna(0, inplace=True)
        local_df.reset_index(inplace=True)
        local_df.sort_values(by=[self.timestamp_id, self.page_id], inplace=True)
        return local_df

    # Getters and setters
    @property
    def session_data(self):
        return self.__session_data

    @session_data.setter
    def session_data(self, new_session_data: pd.DataFrame):
        self.__session_data = new_session_data

    @property
    def page_data(self):
        return self.__page_data

    @page_data.setter
    def page_data(self, new_page_data: pd.DataFrame):
        self.__page_data = new_page_data

    @property
    def page_data_out(self):
        return self.__page_data_out

    @page_data_out.setter
    def page_data_out(self, new_page_data_out: pd.DataFrame):
        self.__page_data_out = new_page_data_out

    @property
    def number_sessions(self):
        return self.session_data.shape[0]

    @property
    def number_pages(self):
        return self.page_data.shape[0]

    @property
    def duration_table(self):
        return self.__duration_table

    @duration_table.setter
    def duration_table(self, new_duration_table: pd.DataFrame):
        self.__duration_table = new_duration_table

    @property
    def search_table(self):
        return self.__search_table

    @search_table.setter
    def search_table(self, new_search_table: pd.DataFrame):
        self.__search_table = new_search_table

    @property
    def kpi_results(self):
        return self.__results

    @kpi_results.setter
    def kpi_results(self, results: dict):
        self.__results = results


class NavigationDataAnalyzer:
    def __init__(self,
                 input_data: pd.DataFrame,
                 metadata: dict,
                 logger_level: int = logging.WARNING):
        self.__logger = logging.Logger(name='default_logger',
                                       level=logger_level)
        self.__input_data = input_data
        self.__metadata = metadata
        self.__data_validator = DataValidator(input_data=input_data,
                                              metadata=metadata,
                                              logger=self.logger)
        self.__session_analyzer = SessionAnalyzer(input_data=input_data,
                                                  metadata=metadata,
                                                  logger=self.logger)

    def get_number_events(self,
                          group_name: str = None):
        """
        Method used to retrieve the number of events in the dataset. It can be also be filtered by group name.
        This function assumes that events are the primary key of the dataset.
        :param group_name: Name of the study groups as defined in metadata (['valid_values']['groups']['valid'])
        :return: Number of events in the dataset (in total or per group)
        """
        groups_id = self.metadata['metadata']['valid_values']['groups']['group_id']
        valid_groups = self.metadata['metadata']['valid_values']['groups']['valid']
        if group_name is None:
            return self.input_data.shape[0]
        else:
            if group_name in valid_groups:
                return self.input_data.loc[self.input_data[groups_id] == group_name].shape[0]
            else:
                self.logger.error(f'{group_name} is not a valid group name. '
                                  f'Please select among those listed here: {", ".join(valid_groups)}')

    def save(self, name: str = 'navigation_data_analyzer.pickle'):
        objects_to_store = dict()
        objects_to_store['metadata'] = self.metadata
        objects_to_store['input_data'] = self.input_data
        objects_to_store['kpi_results'] = self.session_analyzer.kpi_results
        with open(name, 'wb') as fp:
            pickle.dump(objects_to_store, fp)

    @staticmethod
    def load(filepath: str):
        with open(filepath, 'rb') as fp:
            existing_object = pickle.load(fp)
            instance_object = NavigationDataAnalyzer(input_data=existing_object['input_data'],
                                                     metadata=existing_object['metadata'])
            instance_object.session_analyzer.kpi_results = existing_object['kpi_results']
        return instance_object

    def to_excel(self, filename: str):
        excel_writer = pd.ExcelWriter(filename)
        self.session_analyzer.session_data.to_excel(excel_writer, sheet_name='session_data', index=False)
        self.session_analyzer.page_data_out.to_excel(excel_writer, sheet_name='page_data', index=False)
        self.session_analyzer.duration_table.to_excel(excel_writer, sheet_name='duration_table', index=False)
        self.session_analyzer.search_table.to_excel(excel_writer, sheet_name='search_table', index=False)
        for key, value in self.session_analyzer.kpi_results.items():
            results = pd.DataFrame(value)
            results.to_excel(excel_writer, sheet_name=f'kpi_{key}', index=False)
        groups_df = pd.DataFrame({'group': self.session_analyzer.valid_groups})
        groups_df.to_excel(excel_writer, sheet_name='groups', index=False)
        excel_writer.save()
        excel_writer.close()

    # Getters and Setters
    @property
    def session_analyzer(self):
        return self.__session_analyzer

    @property
    def data_validator(self):
        return self.__data_validator

    @property
    def input_data(self):
        return self.__input_data

    @input_data.setter
    def input_data(self, new_input_data: pd.DataFrame):
        self.data_validator.input_data = new_input_data
        self.data_validator.default_pipeline()
        self.__input_data = new_input_data

    @property
    def metadata(self):
        return self.__metadata

    @metadata.setter
    def metadata(self, new_metadata: dict):
        self.__input_data = new_metadata

    @property
    def logger(self):
        return self.__logger

    @logger.setter
    def logger(self, new_logger):
        self.__logger = new_logger
