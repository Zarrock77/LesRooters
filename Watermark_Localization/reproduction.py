"""Bound numerical inference drift without refitting weights or thresholds.

The exact-replay requirement remains the default. These limits apply only to
explicitly requested GPU numerical replay and are checked again before upload.
"""
import numpy as np
from baseline import image_scores

LIMITS = {
    'pixel_disagreement_fraction': 0.001,
    'max_image_pixel_disagreement_fraction': 0.005,
    'probability_mean_absolute_difference': 0.002,
    'absolute_miou_difference': 0.001,
    'max_image_miou_difference': 0.01,
}


def validate_report(report):
    assert report['checked_images'] > 0
    assert 0 <= report['exact_images'] <= report['checked_images']
    for field, limit in LIMITS.items():
        assert np.isfinite(report[field]) and 0 <= report[field] <= limit, (field, report[field], limit)


def compare_prediction(probabilities, threshold, reference_masks, reference_probabilities, truth):
    p=np.asarray(probabilities); q=np.asarray(reference_probabilities)
    a=p>=threshold; b=np.asarray(reference_masks,dtype=bool)
    assert p.shape==q.shape==b.shape==truth.shape and p.ndim==3
    assert np.isfinite(p).all() and np.isfinite(q).all()
    assert ((p>=0)&(p<=1)).all() and ((q>=0)&(q<=1)).all()
    difference=(a!=b).mean(axis=(1,2))
    old=image_scores(b,truth); new=image_scores(a,truth)
    report={'checked_images':len(p),'exact_images':int((difference==0).sum()),
        'pixel_disagreement_fraction':float(difference.mean()),
        'max_image_pixel_disagreement_fraction':float(difference.max()),
        'probability_mean_absolute_difference':float(abs(p-q).mean()),
        'probability_max_absolute_difference':float(abs(p-q).max()),
        'reference_miou':float(old.mean()),'recomputed_miou':float(new.mean()),
        'absolute_miou_difference':float(abs(new.mean()-old.mean())),
        'max_image_miou_difference':float(abs(new-old).max())}
    validate_report(report)
    return report
