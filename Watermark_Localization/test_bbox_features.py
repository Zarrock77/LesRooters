import unittest
import numpy as np
from bbox_features import apply_prior


class BoxPriorTests(unittest.TestCase):
    def test_geometry_and_method_isolation(self):
        rows=[{'image_name':'a','method':'trustmark'},{'image_name':'b','method':'pixelseal'}]
        p=np.full((2,128,128),.3,dtype=np.float32)
        records={'a':[{'box':[128,128,384,384],'score':1.0}]}
        result=apply_prior(p,rows,records,1)
        self.assertEqual(result[0].sum(),64*64)
        self.assertEqual(result[0,32,32],1)
        self.assertEqual(result[0,96,96],0)
        np.testing.assert_array_equal(result[1],p[1])
        np.testing.assert_array_equal(p,np.full_like(p,.3))

    def test_missing_or_low_confidence_preserves_prediction(self):
        rows=[{'image_name':'a','method':'trustmark'},{'image_name':'b','method':'trustmark'}]
        p=np.full((2,128,128),.3,dtype=np.float32)
        records={'a':[],'b':[{'box':[0,0,512,512],'score':.49}]}
        np.testing.assert_array_equal(apply_prior(p,rows,records,1),p)


if __name__=='__main__':unittest.main()
