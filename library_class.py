import util
import pandas as pd
import os

class Library(object):
    
    def load_from_file(self,library_name):
        self.raw_lib = pd.read_parquet(os.path.join('./data',library_name))
    
    def load_from_query(self,query_string,max_results):
        self.raw_lib = util.query_to_df(query_string,max_results)
    
    def clean_library(self):
        
        ## drop columns that we aren't going to modify
        cols = ['title','summary','authors','primary_category','categories']
        input_lib = self.raw_lib[cols].copy()

        input_lib['title'] = input_lib['title'].apply(util.cleanse)
        input_lib['summary'] = input_lib['summary'].apply(util.cleanse)
        input_lib['hyph_in_summary'] = input_lib['summary'].apply(util.find_hyph)
        input_lib['hyph_in_title'] = input_lib['title'].apply(util.find_hyph)
        input_lib['msc_tags'] = input_lib.categories.apply(util.find_msc).apply(util.msc_to_eng)

        self.clean_lib = input_lib