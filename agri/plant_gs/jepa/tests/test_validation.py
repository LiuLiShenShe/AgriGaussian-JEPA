import pytest
import torch

from agri.plant_gs.jepa.validation import gradient_report


def test_gradient_report_validates_trainable_parameters_and_frozen_teacher():
    model = torch.nn.ModuleDict({
        'online': torch.nn.Linear(3, 2),
        'teacher': torch.nn.Linear(3, 2).requires_grad_(False),
    })
    x = torch.tensor([[0.1, -0.2, 0.3]])
    model['online'](x).square().sum().backward()
    report = gradient_report(model)
    assert report['trainable_tensors'] == 2
    assert report['tensors_with_finite_grad'] == 2
    assert report['tensors_without_grad'] == []
    assert report['frozen_tensors'] == 2
    assert report['frozen_tensors_with_grad'] == []
    assert report['global_l2_norm'] > 0
    assert report['branches']['online']['global_l2_norm'] > 0


def test_nonfinite_gradient_is_reported_and_does_not_pass_finite_gate():
    model = torch.nn.Linear(2, 1)
    model(torch.ones(1, 2)).sum().backward()
    model.weight.grad[0, 0] = float('nan')
    report = gradient_report(model)
    assert not report['all_trainable_grads_finite']
    assert len(report['tensors_with_nonfinite_grad']) == 1
    assert report['tensors_with_finite_grad'] == 1
