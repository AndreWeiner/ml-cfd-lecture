#!/usr/bin/env python3
"""Check classification exercise gradients, split, source validation and restore.

Optional --figure-dir should name an initially empty directory. Its comparisons
use the same baseline twice for visual QA, not additional scientific data.
"""
from pathlib import Path
import sys,csv,tempfile,numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'notebooks'))
import classification_exercise_tools as t

def main(figure_dir=None):
    rows=t.load_points(ROOT/'datasets/path_shape_regimes/tripathi2015_digitization/points.csv',lecture=True)
    tr,va=t.split_lecture(rows);assert (len(tr),len(va))==(84,20)
    a=[rows[i] for i in tr];v=[rows[i] for i in va];raw,y=t.arrays(a);rv,yv=t.arrays(v);sc=t.fit_scaler(raw);x=t.transform(raw,sc);xv=t.transform(rv,sc)
    np.testing.assert_allclose(x.min(0),-1);np.testing.assert_allclose(x.max(0),1)
    # Independent moderate-logit CE and all-parameter central differences.
    features=np.array([[-.4,.3],[.7,-.8],[.1,.6]]);labels=np.array([0,3,4]);parameters=t.initialize(width=3,seed=7)
    value,grad=t.loss_and_gradient(features,labels,parameters);hidden=np.tanh(features@parameters[0]+parameters[1]);logits=hidden@parameters[2]+parameters[3];prob=np.exp(logits);prob/=prob.sum(1,keepdims=True)
    np.testing.assert_allclose(value,-np.log(prob[np.arange(3),labels]).mean())
    error=0.
    for p,g in zip(parameters,grad):
     for index in np.ndindex(p.shape):
      old=p[index];h=1e-5;p[index]=old+h;pos=t.loss_and_gradient(features,labels,parameters)[0];p[index]=old-h;neg=t.loss_and_gradient(features,labels,parameters)[0];p[index]=old;error=max(error,abs((pos-neg)/(2*h)-g[index]))
    assert error<1e-8,error
    baseline=t.fit_classifier(x,y,xv,yv);repeat=t.fit_classifier(x,y,xv,yv)
    for key in ['train_loss','validation_loss']:np.testing.assert_array_equal(baseline[key],repeat[key])
    assert baseline['best_epoch']==int(np.argmin(baseline['validation_loss']))
    score=t.metrics(baseline,xv,yv);np.testing.assert_allclose(score['cross_entropy'],min(baseline['validation_loss']));assert score['confusion'].sum()==20
    # A prescribed validation sequence independently exercises patience/restoration.
    saved_lp=t.log_probabilities;saved_gradient=t.loss_and_gradient
    sequence=iter([3.,2.,2.5,2.6]);counter=[0]
    def lp(inputs, params):
     if len(inputs)==2: loss=next(sequence)
     else: loss=1.
     return np.full((len(inputs),5),-loss)
    def grad(inputs, labels, params):return 0.,[np.ones_like(p) for p in params]
    t.log_probabilities=lp;t.loss_and_gradient=grad
    restored=t.fit_classifier(np.zeros((3,2)),np.zeros(3,dtype=int),np.zeros((2,2)),np.zeros(2,dtype=int),epochs=10,patience=2)
    assert restored['best_epoch']==1 and restored['stop_epoch']==3
    for p,initial in zip(restored['parameters'],restored['initial_parameters']):np.testing.assert_allclose(p,initial-.01,atol=1e-8)
    t.log_probabilities=saved_lp;t.loss_and_gradient=saved_gradient
    # Check every parameter in networks with one, two and three hidden layers.
    deep_x=np.array([[-.4,.3],[.7,-.8],[.1,.6],[.2,-.1]])
    deep_y=np.arange(4)
    for sizes in [(3,), (3,2), (3,2,3)]:
        pp=t.initialize(hidden_sizes=sizes,n_classes=4,seed=7)
        _,gg=t.loss_and_gradient(deep_x,deep_y,pp)
        for parameter,gradient in zip(pp,gg):
            for index in np.ndindex(parameter.shape):
                old=parameter[index];step=1e-5
                parameter[index]=old+step;pos=t.loss_and_gradient(deep_x,deep_y,pp)[0]
                parameter[index]=old-step;neg=t.loss_and_gradient(deep_x,deep_y,pp)[0]
                parameter[index]=old
                assert abs((pos-neg)/(2*step)-gradient[index])<1e-8
        diagnostic=t.fit_classifier(deep_x,deep_y,deep_x,deep_y,hidden_sizes=sizes,
                                    n_classes=4,epochs=3,patience=None)
        assert diagnostic['stop_epoch']==3
        assert len(diagnostic['last_parameters'])==2*(len(sizes)+1)
        np.testing.assert_allclose(t.metrics(diagnostic,deep_x,deep_y)['cross_entropy'],
                                   min(diagnostic['validation_loss']))
    # Source metadata fixtures test mechanics, never presented as scientific data.
    fixture=[dict(a[0],source_id='synthetic_schema_fixture',point_id='fixture-1',group_id='fixture-1')]
    enriched=t.merge_training(a,fixture,v);assert len(enriched)==85
    for bad in [[],[a[0]]]:
     try:t.merge_training(a,bad,v)
     except ValueError:pass
     else:raise AssertionError('bad source accepted')
    with tempfile.TemporaryDirectory() as directory:
     p=Path(directory)/'fixture.csv';t.write_points(p,fixture);assert len(t.load_points(p))==1
     p.write_text('point_id,Ga,Eo,regime\nbad,0,2,I\n')
     try:t.load_points(p,lecture=True)
     except ValueError:pass
     else:raise AssertionError('nonpositive input accepted')
     p=Path(directory)/'model.npz';t.save_model(p,baseline,sc);z=np.load(p);np.testing.assert_array_equal(z['log_minimum'],sc[0])
     np.testing.assert_allclose(t.log_probabilities(xv,[z[f'parameter_{j}'] for j in range(4)]),saved_lp(xv,baseline['parameters']))
    # Plot smoke run uses the same baseline twice; this is formatting QA, not new data.
    if figure_dir is not None:
        t.save_comparison(figure_dir,baseline,baseline,a,v,[],sc)
    print('Passed: gradient',error,'; baseline selected/stopped',baseline['best_epoch'],baseline['stop_epoch'],'; validation',score['cross_entropy'],score['errors'])

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--figure-dir", type=Path)
    main(parser.parse_args().figure_dir)
