import unittest
from synthpop import Synthpop
import pandas as pd
import numpy as np

class TestProcessing(unittest.TestCase):

    def test_add_NaN_columns_for_numeric_columns(self):
        df = pd.DataFrame({'a':[1,2,np.nan], 'b':[1,1,1], 'c':['x','y',None]})
        spop = Synthpop()
        dtype_map = {'a':'float','b':'float', 'c':'categorical'}
        res,dtype_res = spop.pre_preprocess(df,dtype_map,nan_fill=-8)

        self.assertTrue('a_NaN' in res,"Nan column not made")
        self.assertFalse('b_NaN' in res,"Nan column should not be made if there are no NaNs")
        self.assertFalse('c_NaN' in res,"Nan column should not be made for categorical columns")
        self.assertTrue(res['a_NaN'][2])
        self.assertEqual(res['a'][2], -8)
        self.assertEqual(dtype_res['a_NaN'],'category')

    def test_apply_and_remove_added_NaN_columns(self):
        df = pd.DataFrame({'a':[1,2,np.nan],'a_NaN':[False,False,True], 'b':[1,1,1], 'c':['x','y',None]})
        spop = Synthpop()


if __name__ == '__main__':
    unittest.main()