"""Safety checks for explicit bounded replay before official submission."""
import unittest
import numpy as np
from reproduction import compare_prediction


class ReplayTests(unittest.TestCase):
    def setUp(self):
        self.reference=np.full((2,128,128),.2)
        self.reference[:,:,:64]=.8
        self.truth=self.reference>=.5

    def compare(self, p):
        return compare_prediction(p,.5,self.truth,self.reference,self.truth)

    def test_exact(self):
        self.assertEqual(self.compare(self.reference)['exact_images'],2)

    def test_tiny_boundary_drift(self):
        p=self.reference.copy(); p[0,0,63]=.49
        report=self.compare(p)
        self.assertEqual(report['exact_images'],1)
        self.assertLess(report['absolute_miou_difference'],.001)

    def test_changed_region_rejected(self):
        p=self.reference.copy(); p[0,:16,:16]=.2
        with self.assertRaises(AssertionError): self.compare(p)

    def test_confidence_drift_rejected_even_with_same_masks(self):
        p=np.where(self.truth,.6,.4)
        with self.assertRaises(AssertionError): self.compare(p)

    def test_empty_class_iou_discontinuity_rejected(self):
        # One pixel can have a large metric effect when an entire class is absent.
        q=np.full((2,128,128),.9); p=q.copy(); p[0,0,0]=.4
        with self.assertRaises(AssertionError): compare_prediction(p,.5,q>.5,q,q>.5)

    def test_nonfinite_rejected(self):
        p=self.reference.copy(); p[0,0,0]=np.nan
        with self.assertRaises(AssertionError): self.compare(p)


if __name__=='__main__': unittest.main()
