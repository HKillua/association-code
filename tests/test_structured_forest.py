import itertools
import numpy as np
import unittest
import torch
from association.training.structured_forest import maximum_forest,forest_contrast,partial_fy_loss
from tools.analysis.object_forest_qualification import object_rows,point_coverage


def test_exact_unconstrained_and_full_label_optima():
    p=np.column_stack(np.triu_indices(4,1));s=np.array([1.,8.,3.,2.,7.,4.]);labels=np.array([0,0,1,1])
    for constrained in [None,labels]:
        values=[]
        for edges in itertools.combinations(range(6),2):
            # Two distinct edges cannot form a cycle in a simple N4 graph.
            if constrained is None or all(labels[p[e,0]]==labels[p[e,1]] for e in edges):values.append(s[list(edges)].sum())
        actual=maximum_forest(s,p,4,2,constrained)
        assert actual.sum()==2 and s[actual].sum()==max(values)
    # Rank3: three edges of a triangle must be rejected as cyclic.
    a=maximum_forest(np.array([10.,9.,0.,8.,0.,1.]),p,4,1)
    assert a.sum()==3 and not a[[0,1,3]].all()


def test_envelope_gradient_translation_and_permutation():
    p=np.column_stack(np.triu_indices(4,1));s=np.array([1.,8.,3.,2.,7.,4.]);labels=np.array([0,0,1,1]);z=np.array([[.02,.01,.04,-.02,.03,.06]])
    x=torch.tensor(s,requires_grad=True);loss=partial_fy_loss(x,p,labels,z);loss.backward()
    for i in range(6):
        t=s.copy();t[i]+=1e-5;co,o=forest_contrast(t,p,labels,z);v=co@t+o
        assert abs((v-loss.item())/1e-5-x.grad[i].item())<1e-7
    assert loss.item()>=0 and abs(x.grad.sum().item())<1e-12
    assert abs(partial_fy_loss(x+17,p,labels,z).item()-loss.item())<1e-12
    order=np.array([2,0,3,1]);mat=np.zeros((4,4));mat[p[:,0],p[:,1]]=s;mat+=mat.T
    old=maximum_forest(s,p,4,2);new=maximum_forest(mat[np.ix_(order,order)][p[:,0],p[:,1]],p,4,2)
    oldedges={tuple(e) for e in p[old]};newedges={tuple(sorted(order[e])) for e in p[new]};assert oldedges==newedges
    assert partial_fy_loss(x,p,np.arange(4),z).item()==0
    with unittest.TestCase().assertRaises(ValueError):maximum_forest(s,p,4,3,labels)


def test_all_object_losses_include_singleton_and_retained_controls():
    ids=np.array(['a','b','b']);p=np.array([[0,1],[0,2],[1,2]])
    rows=object_rows(ids,np.array(['a','b']),np.array([0,1,1]),np.array([0,0,0]),np.array([True,False,False]),p,np.ones((2,2),bool),np.array([[1,1],[0,0]],bool))
    assert len(rows)==2 and rows[1]['lost'] and rows[1]['fragments']==2 and rows[1]['forest_false_edges_incident']==1
    assert rows[1]['lost_points']==2 and rows[0]['positive_pairs']==0 and rows[0]['represented_after']


class ForestTests(unittest.TestCase):
    def test_optimum(self):test_exact_unconstrained_and_full_label_optima()
    def test_gradient(self):test_envelope_gradient_translation_and_permutation()
    def test_objects(self):test_all_object_losses_include_singleton_and_retained_controls()
    def test_unmatched_rows_have_no_eligible_reference(self):
        a=dict(X_aligned_target_ids=np.array(['a','']),X_aligned_prediction_valid=np.ones((2,2,4),bool),X_aligned_truth_valid=np.array([np.ones((2,4),bool),np.zeros((2,4),bool)]))
        self.assertTrue(point_coverage(a,'X',['a']).all())
        a['X_aligned_truth_valid'][1]=True
        with self.assertRaises(ValueError):point_coverage(a,'X',['a'])

if __name__=='__main__':unittest.main()
