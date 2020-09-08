"""
This code creates .docx files with generated summaries for comparing manually
"""
from src.analysis_tools import read_data, create_dfs_info, process_topics

path_to_save = '../data/Prepared_for_check/'
path_to_read = '../data/analyzed_data/'

possible_types = ['Tech', 'Humanity', 'Science'] #types of texts under analysis

res_dict, res_dfs = read_data(path_to_read)
create_dfs_info(res_dfs, path_to_save + 'Info/', possible_types)
process_topics(res_dict, path_to_save)