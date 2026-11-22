"""Export the strongest robust CV candidate, averaging its five trained folds.

Candidate eligibility: positive gain over the refitted baseline in every fold
and positive lower paired bootstrap bound. Selection never reads public scores.
The CV score evaluates OOF constituent models; it is not a direct measurement of
the five-fold ensemble applied to unseen test images.
"""
import argparse
import gc
import json
import lzma
import os
from pathlib import Path
import sys
import numpy as np
import torch
import localization_v2 as experiment
from baseline import METHODS, write_json, image_scores
from compare_cv import calibrated_probability
from prepare_submission import sha256, validate_payload
from reproduction import LIMITS, compare_prediction


def verify_external_provenance(kit_root):
    """Bind external CV to the exact external-only initialization and code."""
    from run_external_cv import SOURCE_FILES, guidance_provenance, verify_external_pretraining
    root=Path(kit_root)/'output'
    protocol_path=root/'cv_external_v1/protocol.json'
    protocol=json.loads(protocol_path.read_text())
    assert protocol['architecture']=='external'
    assert protocol['epochs']==120 and protocol['seed']==20261123
    checkpoint,bundle=verify_external_pretraining(kit_root)
    for key,value in bundle.items():
        assert protocol[key]==value, ('External initialization changed after CV',key)
    del checkpoint
    code=Path(__file__).resolve().parent
    assert set(protocol['source_sha256'])==set(SOURCE_FILES)
    for name in SOURCE_FILES:
        assert protocol['source_sha256'][name]==sha256(code/name), ('External CV source changed',name)
    assert protocol['guidance_cache_sha256']==guidance_provenance(Path(kit_root)), 'External CV guidance changed'
    reference=root/'cv_adobe_v1'
    assert protocol['reference_protocol_sha256']==sha256(reference/'protocol.json')
    assert protocol['folds']==json.loads((reference/'protocol.json').read_text())['folds']
    for fold in range(5):
        assert protocol['reference_inner_split_sha256'][str(fold)]==sha256(reference/f'fold_{fold}/inner_split.json')
        assert json.loads((root/f'cv_external_v1/fold_{fold}/inner_split.json').read_text())==json.loads((reference/f'fold_{fold}/inner_split.json').read_text())
    assert protocol['metadata_sha256']==sha256(Path(kit_root)/'release/train/metadata.json')
    assert protocol['masks_sha256']==sha256(Path(kit_root)/'release/train/masks_128.npz')
    return {'protocol':protocol,'protocol_sha256':sha256(protocol_path),
            'pretraining_manifest':bundle['external_pretraining_manifest'],
            'pretraining_manifest_sha256':bundle['external_pretraining_manifest_sha256'],
            'pretraining_protocol_sha256':bundle['external_pretraining_protocol_sha256'],
            'checkpoint_sha256':bundle['external_checkpoint_sha256'],
            'checkpoint_provenance':bundle['external_checkpoint_provenance'],
            'pixel_encoder_sha256':bundle['external_pixel_encoder_sha256'],
            'generation_manifest_sha256':bundle['external_data_manifest_sha256'],
            'generation_spec_sha256':bundle['external_generation_spec_sha256'],
            'clean_corpus_manifest_sha256':bundle['external_clean_corpus_manifest_sha256']}


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--kit-root',type=Path,required=True)
    ap.add_argument('--guidance-root',type=Path,required=True)
    ap.add_argument('--comparison-root',type=Path,required=True)
    ap.add_argument('--output-root',type=Path,required=True)
    ap.add_argument('--version',default='cv_ensemble_v1')
    ap.add_argument('--architecture',choices=('residual','pretrained','finetuned','pretrained_tta','detector','adobe','spatial','synthetic','external','equal_blend','detector_blend'),
                    help='Export this eligible candidate instead of selecting the highest eligible OOF score.')
    ap.add_argument('--allow-numerical-replay',action='store_true',help='Require bounded, recorded GPU drift instead of exact equality.')
    args=ap.parse_args(); args.data_root=args.kit_root/'release'
    sys.path.insert(0,str(args.kit_root)); torch.set_num_threads(4); torch.backends.cudnn.benchmark=True
    args.output_root.mkdir(parents=True,exist_ok=True)
    report=json.loads((args.comparison_root/'metrics.json').read_text())
    eligible={k:v for k,v in report['candidates'].items() if min(v['fold_gain'])>0 and v.get('paired_gain_99pct',v.get('paired_gain_95pct'))[0]>0}
    decision_path=args.comparison_root/'replacement_decision.json'
    if decision_path.exists():
        assert json.loads(decision_path.read_text())['export_recommended'],'Candidate did not pass the frozen promotion rule.'
    assert eligible, 'No robust improvement over baseline; do not export.'
    choice=args.architecture or max(eligible,key=lambda k:eligible[k]['oof_miou'])
    assert choice in eligible, 'Requested candidate did not pass the paired-fold eligibility checks.'
    if choice=='spatial':
        assert decision_path.exists(), 'Spatial export requires the frozen promotion-rule decision.'
        decision=json.loads(decision_path.read_text())
        assert decision['candidate']=='spatial' and decision['reference']=='adobe', 'Spatial must be compared against the current Adobe incumbent.'
    if choice=='synthetic':
        assert decision_path.exists(), 'Synthetic export requires the frozen promotion-rule decision.'
        decision=json.loads(decision_path.read_text())
        spatial_decision_path=args.kit_root/'output/cv_spatial_v1/replacement_decision.json'
        assert spatial_decision_path.exists(), 'Assess the prior spatial experiment before synthetic promotion.'
        spatial_decision=json.loads(spatial_decision_path.read_text())
        assert spatial_decision['candidate']=='spatial' and spatial_decision['reference']=='adobe'
        expected_reference='spatial' if spatial_decision['export_recommended'] else 'adobe'
        assert decision['candidate']=='synthetic' and decision['reference']==expected_reference, 'Synthetic comparison is not against the current validated incumbent.'
    external_provenance=None
    if choice=='external':
        from finish_external import prior_external_decisions
        assert decision_path.exists(), 'External export requires the frozen promotion-rule decision.'
        decision=json.loads(decision_path.read_text())
        expected_reference,prior_decisions,prior_decision_hashes=prior_external_decisions(args.kit_root)
        assert decision['candidate']=='external' and decision['reference']==expected_reference, 'External comparison is not against the strongest prior validated incumbent.'
        external_provenance=verify_external_provenance(args.kit_root)
    architectures={'equal_blend':('residual','pretrained'),'detector_blend':('pretrained','detector')}.get(choice,(choice,))
    rows=json.loads((args.data_root/'train/metadata.json').read_text())
    with np.load(args.data_root/'train/masks_128.npz') as z:
        truth=dict(zip(z['names'].tolist(),z['masks'].astype(bool)))
    test=json.loads((args.data_root/'test/metadata.json').read_text()); names=[r['image_name'] for r in test]
    assert not set(names)&{r['image_name'] for r in rows}
    guide_args=argparse.Namespace(kit_root=args.kit_root,data_root=args.data_root,output_root=args.guidance_root)
    train_guides=experiment.guidance(guide_args,rows,'train'); test_guides=experiment.guidance(guide_args,test,'test')
    extended_train=extended_test=None
    if 'detector' in architectures:
        from detector_segmenter import detection_guidance
        extended_train=detection_guidance(guide_args,rows,'train');extended_test=detection_guidance(guide_args,test,'test')
    train_images={r['image_name']:experiment.read_image(args.data_root,'train',r['image_name']) for r in rows}
    test_images={r['image_name']:experiment.read_image(args.data_root,'test',r['image_name']) for r in test}
    sums={n:np.zeros((128,128),dtype=np.float32) for n in names}; verified=0; exact=0; components=[]
    for architecture in architectures:
        inference=experiment.predict
        train_maps,test_maps=train_guides,test_guides
        if architecture in ('pretrained','finetuned','pretrained_tta'):
            from pretrained_segmenter import PretrainedSegmenter
            model=PretrainedSegmenter().cuda()
            if architecture=='pretrained_tta':
                from dihedral import predict_dihedral
                inference=predict_dihedral
        elif architecture=='detector':
            from detector_segmenter import DetectorSegmenter
            model=DetectorSegmenter().cuda();train_maps,test_maps=extended_train,extended_test
        elif architecture=='adobe':
            from adobe_segmenter import AdobeSegmenter
            model=AdobeSegmenter(args.kit_root).cuda()
        elif architecture in ('spatial','synthetic','external'):
            from spatial_segmenter import SpatialSegmenter
            model=SpatialSegmenter(args.kit_root).cuda()
        else: model=experiment.ResidualSegmenter().cuda()
        model_root=args.kit_root/'output'/('cv_'+architecture+'_v1')
        for fold in range(5):
            folder=model_root/('fold_'+str(fold))
            if architecture=='external':
                initialization=json.loads((folder/'initialization.json').read_text())
                assert initialization['external_checkpoint_sha256']==external_provenance['checkpoint_sha256']
                assert initialization['external_pretraining_protocol_sha256']==external_provenance['pretraining_protocol_sha256']
                assert initialization['strict_spatial_restore'] is True and initialization['pixel_encoder_frozen'] is True
                assert initialization['pixel_encoder_sha256']==external_provenance['pixel_encoder_sha256']
            with np.load(folder/'validation.npz') as z:
                reference=dict(zip(z['names'].tolist(),z['masks']))
                reference_prob=dict(zip(z['names'].tolist(),z['probabilities']))
            split=json.loads((folder/'inner_split.json').read_text())
            assert set(reference)==set(split['validation'])
            assert not (set(split['fit'])|set(split['tune']))&set(reference)
            assert len(reference)==102
            for method in METHODS:
                path=folder/('best_'+method+'.pt'); ckpt=torch.load(path,weights_only=True,map_location='cpu')
                if architecture=='external':
                    from run_external_cv import pixel_state_digest
                    assert pixel_state_digest(ckpt['state_dict'])==external_provenance['pixel_encoder_sha256'], 'Real fold modified the frozen pretrained PixelSeal encoder'
                model.load_state_dict(ckpt['state_dict']); threshold=ckpt['selection']['threshold']
                validation=[r for r in rows if r['image_name'] in reference and r['method']==method]
                p=inference(model,experiment.Images(validation,train_images,train_maps))
                outer_miou=float(image_scores(p>=threshold,np.stack([truth[r['image_name']] for r in validation])).mean())
                replay=compare_prediction(p,threshold,np.stack([reference[r['image_name']] for r in validation]),
                    np.stack([reference_prob[r['image_name']] for r in validation]),
                    np.stack([truth[r['image_name']] for r in validation]))
                if not args.allow_numerical_replay:
                    assert replay['exact_images']==replay['checked_images'],'Validation masks differ; inspect numerical replay before proceeding.'
                verified+=replay['checked_images']; exact+=replay['exact_images']
                fit=[r for r in rows if r['image_name'] in split['fit'] and r['method']==method]
                fit_prob=inference(model,experiment.Images(fit,train_images,train_maps))
                fit_miou=float(image_scores(fit_prob>=threshold,np.stack([truth[r['image_name']] for r in fit])).mean())
                subset=[r for r in test if r['method']==method]
                p=inference(model,experiment.Images(subset,test_images,test_maps))
                for row,prob in zip(subset,p): sums[row['image_name']]+=calibrated_probability(prob,threshold)
                components.append({'architecture':architecture,'fold':fold,'method':method,
                                   'selection':ckpt['selection'],'checkpoint_sha256':sha256(path),
                                   'validation_replay':replay,
                                   'fit_miou':fit_miou,'outer_miou':outer_miou,
                                   'fit_minus_outer_miou':fit_miou-outer_miou})
            print('ENSEMBLE_COMPONENT',architecture,fold,'verified',verified,flush=True)
        del model; gc.collect(); torch.cuda.empty_cache()
    expected=510*len(architectures); assert verified==expected
    masks=np.stack([sums[n]/(5*len(architectures))>=.5 for n in names])
    assert args.version.replace('_','').isalnum()
    artifact=args.output_root/('lesrooters_watermark_'+args.version+'.pt')
    torch.save({'image_names':names,'masks':torch.from_numpy(masks).bool()},artifact)
    checked=validate_payload(artifact,names)
    manifest={'artifact':artifact.name,'sha256':sha256(artifact),'bytes':artifact.stat().st_size,
              'task_id':'34-watermark-localisation-image','images':1020,'shape':list(checked.shape),'dtype':str(checked.dtype),
              'binary':True,'unique_names_match_test_metadata':True,'validation_expected_predictions':expected,
              'validation_predictions_reproduced_exactly':exact,'validation_predictions_checked':verified,
              'validation_replay_mode':'exact' if exact==verified else 'bounded_gpu_numerics',
              'validation_replay_limits':LIMITS,'validation_reference':'five disjoint outer partitions per architecture; fitted models share training images',
              'candidate':choice,'candidate_oof_metrics':eligible[choice],'components':components,
              'ensemble_rule':'Equal mean of threshold-calibrated probabilities over five folds and selected architecture(s), threshold 0.5.',
              'evaluation_caveat':'OOF metric evaluates held-out constituent models, not the final multi-fold test ensemble. Architectures explored before CV.',
              'thresholds_refitted':False,'comparison_sha256':sha256(args.comparison_root/'metrics.json'),
              'test_metadata_sha256':sha256(args.data_root/'test/metadata.json'),'export_script_sha256':sha256(__file__),
              'slurm_job_id':os.getenv('SLURM_JOB_ID'),'uploaded':False}
    if any(architecture in architectures for architecture in ('adobe','spatial','synthetic','external')):
        manifest['adobe_provenance']=json.loads((args.kit_root/'models/trustmark_bbox/provenance.json').read_text())
        manifest['adobe_segmenter_sha256']=sha256(Path(__file__).with_name('adobe_segmenter.py'))
    if 'adobe' in architectures:
        manifest['learned_extension']=json.loads(Path(__file__).with_name('adobe_experiment.json').read_text())
    if 'spatial' in architectures:
        protocol_path=args.kit_root/'output/cv_spatial_v1/protocol.json'
        manifest['learned_extension']=json.loads(protocol_path.read_text())
        manifest['spatial_protocol_sha256']=sha256(protocol_path)
        manifest['spatial_segmenter_sha256']=sha256(Path(__file__).with_name('spatial_segmenter.py'))
        manifest['pixelseal_checkpoint_sha256']=sha256(args.kit_root/'models/pixelseal/checkpoint.pth')
        manifest['pixelseal_model_card_sha256']=sha256(args.kit_root/'models/pixelseal_card.yaml')
        manifest['replacement_decision']=json.loads(decision_path.read_text())
        manifest['replacement_decision_sha256']=sha256(decision_path)
    if 'synthetic' in architectures:
        protocol_path=args.kit_root/'output/cv_synthetic_v1/protocol.json'
        manifest['learned_extension']=json.loads(protocol_path.read_text())
        manifest['synthetic_protocol_sha256']=sha256(protocol_path)
        manifest['spatial_segmenter_sha256']=sha256(Path(__file__).with_name('spatial_segmenter.py'))
        manifest['dihedral_guidance_sha256']=sha256(Path(__file__).with_name('dihedral_guidance.py'))
        manifest['pixelseal_checkpoint_sha256']=sha256(args.kit_root/'models/pixelseal/checkpoint.pth')
        manifest['pixelseal_model_card_sha256']=sha256(args.kit_root/'models/pixelseal_card.yaml')
        synthetic_root=args.kit_root/'output/synthetic_v1'
        manifest['synthetic_generation_spec_sha256']=sha256(synthetic_root/'spec.json')
        manifest['synthetic_generation_manifest_sha256']=sha256(synthetic_root/'manifest.json')
        assert manifest['learned_extension']['synthetic_manifest_sha256']==manifest['synthetic_generation_manifest_sha256']
        assert manifest['learned_extension']['synthetic_manifest']['spec_sha256']==manifest['synthetic_generation_spec_sha256']
        manifest['replacement_decision']=decision
        manifest['replacement_decision_sha256']=sha256(decision_path)
        manifest['prior_spatial_decision']=spatial_decision
        manifest['prior_spatial_decision_sha256']=sha256(spatial_decision_path)
    if 'external' in architectures:
        manifest['learned_extension']=external_provenance['protocol']
        manifest['external_protocol_sha256']=external_provenance['protocol_sha256']
        manifest['external_pretraining_manifest']=external_provenance['pretraining_manifest']
        manifest['external_pretraining_manifest_sha256']=external_provenance['pretraining_manifest_sha256']
        manifest['external_pretraining_protocol_sha256']=external_provenance['pretraining_protocol_sha256']
        manifest['external_checkpoint_sha256']=external_provenance['checkpoint_sha256']
        manifest['external_checkpoint_provenance']=external_provenance['checkpoint_provenance']
        manifest['external_pixel_encoder_sha256']=external_provenance['pixel_encoder_sha256']
        manifest['external_generation_manifest_sha256']=external_provenance['generation_manifest_sha256']
        manifest['external_generation_spec_sha256']=external_provenance['generation_spec_sha256']
        manifest['external_clean_corpus_manifest_sha256']=external_provenance['clean_corpus_manifest_sha256']
        manifest['spatial_segmenter_sha256']=sha256(Path(__file__).with_name('spatial_segmenter.py'))
        manifest['dihedral_guidance_sha256']=sha256(Path(__file__).with_name('dihedral_guidance.py'))
        manifest['pixelseal_checkpoint_sha256']=sha256(args.kit_root/'models/pixelseal/checkpoint.pth')
        manifest['pixelseal_model_card_sha256']=sha256(args.kit_root/'models/pixelseal_card.yaml')
        manifest['replacement_decision']=decision
        manifest['replacement_decision_sha256']=sha256(decision_path)
        manifest['prior_decisions']=prior_decisions
        manifest['prior_decision_sha256']=prior_decision_hashes
    write_json(args.output_root/'manifest.json',manifest)
    with lzma.open(str(artifact)+'.xz','wb',preset=6) as f:f.write(artifact.read_bytes())
    print('READY',choice,artifact.name,sha256(artifact),flush=True)


if __name__=='__main__':main()
